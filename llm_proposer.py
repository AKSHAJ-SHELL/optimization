"""LLM proposal distribution for packer.py (--proposer llm).

Loads a (fine-tuned) code model via MLX and samples discrete moves from the
canonical-state prompt. Invalid generations fall back to None, which
worker_elite replaces with a random move -- so the loop never stalls on a
misbehaving model, and the fallback rate is itself a useful metric (printed
at exit).

Usage inside packer:  --proposer llm --model ./fused_model_or_hf_repo
"""

import atexit

import moves as moves_mod


class LLMProposer:
    def __init__(self, model_path, temperature=0.8, max_tokens=80):
        if model_path is None:
            raise SystemExit("--proposer llm requires --model")
        try:
            from mlx_lm import load, generate  # noqa: F401
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            raise SystemExit("pip install mlx-lm  (Apple Silicon required)")
        self._generate = generate
        self._sampler = make_sampler(temp=temperature)
        # accept either a full model dir/repo OR an mlx_lm adapter dir
        # (adapters_sft/): resolve the base model from adapter_config.json
        import os as _os
        import json as _json
        adapter_path = None
        acfg = _os.path.join(model_path, "adapter_config.json")
        if _os.path.isdir(model_path) and _os.path.exists(acfg):
            with open(acfg) as f:
                base = _json.load(f).get("model")
            if base:
                adapter_path = model_path
                model_path = base
            else:
                raise SystemExit(
                    f"{acfg} has no 'model' key; fuse instead: "
                    "mlx_lm.fuse --model <base> --adapter-path "
                    f"{model_path} --save-path model_sft")
        if adapter_path:
            self.model, self.tokenizer = load(model_path,
                                              adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = load(model_path)
        self.n_calls = 0
        self.n_fallback = 0
        self.max_tokens = max_tokens
        atexit.register(self._report)

    def _report(self):
        if self.n_calls:
            rate = 100.0 * self.n_fallback / self.n_calls
            print(f"[llm_proposer] {self.n_calls} samples, "
                  f"{rate:.1f}% invalid -> random fallback")

    def propose(self, state, k):
        """Return k moves (canonical indices); invalid slots are None."""
        prompt = moves_mod.render_prompt(state)
        if self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        out = []
        for _ in range(k):
            self.n_calls += 1
            text = self._generate(self.model, self.tokenizer, prompt=prompt,
                                  max_tokens=self.max_tokens,
                                  sampler=self._sampler, verbose=False)
            mv = moves_mod.parse_move_json(text, state["n"])
            if mv is None:
                self.n_fallback += 1
            out.append(mv)
        return out
