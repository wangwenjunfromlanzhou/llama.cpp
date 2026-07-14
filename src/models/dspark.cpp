#include "models.h"

// DSpark = DFlash backbone + a semi-autoregressive Markov head applied in-graph by the decoder

void llama_model_dspark::load_arch_hparams(llama_model_loader & ml) {
    llama_model_dflash::load_arch_hparams(ml);

    // ponytail: dflash forces swa_type=STANDARD unconditionally; Gemma4 DSpark
    // draft has no SWA layers, so the SWA memory path asserts. Reset to NONE.
    hparams.swa_type = LLAMA_SWA_TYPE_NONE;

    ml.get_key(LLM_KV_BLOCK_SIZE, hparams.n_dspark_block, /*required*/ true);

    // Gemma4 DSpark ships final_logit_softcapping in its config; Qwen DSpark
    // does not, so override the Gemma-family hparam default and leave softcap disabled.
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // Gemma4 DSpark uses GELU+PAR (GEGLU); Qwen3 DSpark uses SiLU+PAR (SwiGLU).
    // Gate via hidden_activation metadata so the dflash graph picks the right op.
    std::string hidden_act = "silu";
    ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false);
    hparams.f_dspark_ffn_gelu = (hidden_act == "gelu" || hidden_act == "geglu");

    // Gemma family applies attn_logit_softcapping (=50.0) to QK; Qwen DSpark does not.
    // hparams.f_attn_logit_softcapping defaults to 50.0 already; only flip the gate on.
    const bool has_attn_cap = ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
    hparams.attn_soft_cap = has_attn_cap;
}

void llama_model_dspark::load_arch_tensors(llama_model_loader & ml) {
    llama_model_dflash::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    uint32_t markov_rank = 0;
    ml.get_key(LLM_KV_MARKOV_RANK, markov_rank, /*required*/ true);
    const int64_t R = (int64_t) markov_rank;

    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { R, n_vocab }, 0);
    dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { R, n_vocab }, 0);
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + R, 1 }, TENSOR_NOT_REQUIRED);
    dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_dspark::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

// DSpark encoder == DFlash encoder
template <>
llama_model_dspark::graph<true>::graph(const llama_model & model, const llm_graph_params & params)
    : llama_model_dflash::graph<true>(model, params) {}

// anchor (committed last) token of every draft block: token 0 of each block in the ubatch
class llm_graph_input_dspark_anchor : public llm_graph_input_i {
public:
    llm_graph_input_dspark_anchor(uint32_t block_size) : block_size(block_size) {}
    virtual ~llm_graph_input_dspark_anchor() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token);
        const int64_t n_blocks = anchors->ne[0];
        std::vector<int32_t> buf(n_blocks);
        for (int64_t j = 0; j < n_blocks; ++j) {
            buf[j] = ubatch->token[j*block_size];
        }
        ggml_backend_tensor_set(anchors, buf.data(), 0, n_blocks*sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.token && anchors &&
               anchors->ne[0]*(int64_t) block_size == (int64_t) params.ubatch.n_tokens;
    }

    ggml_tensor * anchors = nullptr; // I32 [n_blocks]

    const uint32_t block_size;
};

// DSpark decoder: DFlash decoder + Markov bias on the draft logits, chained per block position:
//   logits'(i) = logits(i) + markov_w2 . markov_w1[prev(i)]
//   prev(0)    = the block's anchor token, prev(i>0) = argmax(logits'(i-1))
template <>
llama_model_dspark::graph<false>::graph(const llama_model & model, const llm_graph_params & params)
    : llama_model_dflash::graph<false>(model, params) {
    // KV-injection (embd) batch: no logits to bias
    if (ubatch.embd) {
        return;
    }

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark markov weights not loaded");

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    // Gemma4 ships a final_logit_softcapping; the base lm_head logits must be
    // softcapped (scale(1/cap) -> tanh -> scale(cap)) BEFORE the Markov bias is
    // added -- matching Gemma4 DSpark's LogitsProcessor + Markov head ordering.
    const float cap = model.hparams.f_final_logit_softcapping;
    if (cap > 0.0f) {
        base = ggml_scale(ctx0, base, 1.0f / cap);
        base = ggml_tanh(ctx0, base);
        base = ggml_scale(ctx0, base, cap);
    }

    const int64_t bs = model.hparams.n_dspark_block;
    GGML_ASSERT(bs > 0);

    // the drafting loop always submits whole anchor-first blocks
    if (n_tok % bs != 0) {
        return;
    }
    const int64_t n_blocks = n_tok / bs;

    auto inp = std::make_unique<llm_graph_input_dspark_anchor>((uint32_t) bs);
    inp->anchors = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blocks);
    ggml_set_input(inp->anchors);
    ggml_tensor * prev = inp->anchors; // I32 [n_blocks]
    res->add_input(std::move(inp));

    // optional confidence head: per-position predicted acceptance probability.
    // conf_i = sigmoid(conf_proj . concat(h_i, markov_w1[prev_i]) + bias)
    // h_i is res->t_embd (pre-lm_head hidden state, set by the dflash graph).
    ggml_tensor * conf_proj = model.dspark_conf_proj;
    ggml_tensor * h_full    = (conf_proj != nullptr) ? res->t_embd : nullptr; // [n_embd, n_tokens]
    ggml_tensor * conf_cat  = nullptr;

    ggml_tensor * cat = nullptr;
    for (int64_t i = 0; i < bs; ++i) {
        ggml_tensor * markov_emb = ggml_get_rows(ctx0, w1, prev); // [R, n_blocks]
        ggml_tensor * bias       = ggml_mul_mat(ctx0, w2, markov_emb); // [n_vocab, n_blocks]

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, bs*base->nb[1], i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        if (h_full != nullptr) {
            // h at position i of each block: [n_embd, n_blocks] strided view -> cont
            // (concat over a strided view is fragile; wrap in cont)
            ggml_tensor * h_i   = ggml_view_2d(ctx0, h_full, n_embd, n_blocks, bs*h_full->nb[1], i*h_full->nb[1]);
            ggml_tensor * cat_i = ggml_concat(ctx0, ggml_cont(ctx0, h_i), markov_emb, 0); // [n_embd+R, n_blocks]
            ggml_tensor * logit = ggml_mul_mat(ctx0, conf_proj, cat_i);  // [1, n_blocks]
            if (model.dspark_conf_proj_b != nullptr) {
                logit = ggml_add(ctx0, logit, model.dspark_conf_proj_b);
            }
            ggml_tensor * conf_i = ggml_sigmoid(ctx0, logit); // [1, n_blocks]
            conf_cat = conf_cat ? ggml_concat(ctx0, conf_cat, conf_i, 1) : conf_i;
        }

        if (i + 1 < bs) {
            prev = ggml_argmax(ctx0, col); // I32 [n_blocks]
        }
    }

    // cat is position-major; restore the ubatch's block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, bs);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, bs, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    res->t_logits = out;
    ggml_build_forward_expand(gf, out);

    // emit confidence in the same block-major order, then broadcast to t_embd's
    // shape and route through t_h_nextn so the existing embeddings_nextn extract
    // path picks it up (a plain [1, n_tok] output tensor fails backend alloc).
    if (conf_cat != nullptr) {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, conf_cat, 1, n_blocks, bs);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3)); // [1, bs, n_blocks]
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);
        conf = ggml_repeat(ctx0, conf, res->t_embd); // [n_embd, n_tok]
        res->t_h_nextn = conf;
        ggml_build_forward_expand(gf, conf);
    }
}
