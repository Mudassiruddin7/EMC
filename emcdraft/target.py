"""Thin wrapper around the Hugging Face implementation of the target model (offline use only).

It returns the final, post-norm hidden state of every position - exactly the tensor the engine
hands to the MTP head - for whole sequences in one teacher-forced forward pass. Sequences in a
batch are right-padded: in a causal model, padding *after* a sequence cannot change its hidden
states, so no attention mask and no position bookkeeping is needed.
"""
from __future__ import annotations

import torch

PAD_ID = 248044


def load_model(path: str):
    import transformers
    errs = []
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        for dk in ("dtype", "torch_dtype"):
            try:
                model = cls.from_pretrained(path, device_map={"": 0}, **{dk: torch.bfloat16})
                if next(model.parameters()).dtype != torch.bfloat16:
                    raise RuntimeError("model was not loaded in bfloat16")
                return model.eval()
            except Exception as e:  # noqa: BLE001 - try the next class / spelling
                errs.append(f"{name}({dk}): {type(e).__name__}: {str(e)[:300]}")
                torch.cuda.empty_cache()
    raise RuntimeError("could not load the target model:\n  " + "\n  ".join(errs))


class Target:
    def __init__(self, path: str):
        self.model = load_model(path)
        head = self.model.get_output_embeddings()
        self.W = head.weight
        self.E = self.model.get_input_embeddings().weight
        self.device = self.W.device
        self._cap = None
        head.register_forward_pre_hook(self._hook)
        self.mode = self._select_mode()

    def _hook(self, _mod, args):
        self._cap = args[0]

    def _forward(self, ids, mode):
        if mode == "base":
            return self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        out = self.model(input_ids=ids, use_cache=False, output_hidden_states=True, logits_to_keep=1)
        return out.hidden_states[-1]

    @torch.no_grad()
    def _select_mode(self) -> str:
        """Pick the call whose output equals the tensor the LM head consumes."""
        ids = torch.tensor([[248045, 846, 198, 6820, 488, 513, 264, 5484, 6618, 22839, 13]] * 2,
                           device=self.device)
        self.model(input_ids=ids, use_cache=False)
        ref = self._cap.float()
        for mode in ("base", "states"):
            try:
                h = self._forward(ids, mode).float()
                if h.shape == ref.shape and float((h - ref).abs().max()) <= 0.02 * float(ref.abs().max()):
                    return mode
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError("could not obtain the final hidden states from the target model")

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        return self._forward(ids, self.mode)
