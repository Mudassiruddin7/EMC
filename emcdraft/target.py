"""Thin wrapper around the Hugging Face implementation of the target model.

Only used offline (data generation and hidden-state extraction); serving always goes through
the pinned engine. The wrapper exposes the *final, post-norm* hidden state of every position,
which is exactly what the engine hands to the MTP head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

STOP_IDS = (248046, 248044)          # <|im_end|>, <|endoftext|>  (data/dev/meta.json)
CHAT_PREFIX = (248045, 846, 198)     # <|im_start|>user\n
CHAT_SUFFIX = (248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271)
#             <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n   (enable_thinking=False)


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
            except Exception as e:  # noqa: BLE001 - try the next spelling / class
                errs.append(f"{name}({dk}): {type(e).__name__}: {str(e)[:300]}")
                torch.cuda.empty_cache()
    raise RuntimeError("could not load the target model:\n  " + "\n  ".join(errs))


class Target:
    def __init__(self, path: str):
        self.model = load_model(path)
        self.lm_head = self.model.get_output_embeddings()
        self.embed = self.model.get_input_embeddings()
        self.W = self.lm_head.weight
        self.device = self.W.device
        self.base = getattr(self.model, "model", None)
        self._cap = None
        self.lm_head.register_forward_pre_hook(self._hook)
        self.use_base = self._check_base()

    def _hook(self, _mod, args):
        self._cap = args[0]

    @torch.no_grad()
    def _check_base(self) -> bool:
        """Is `model.model(...)` usable and does it return the tensor fed to the LM head?"""
        ids = torch.tensor([list(CHAT_PREFIX) + [6820, 488, 513, 264, 5484]], device=self.device)
        self.model(input_ids=ids)
        ref = self._cap.float()
        try:
            out = self.base(input_ids=ids, use_cache=True)
            h = out.last_hidden_state.float()
            return bool(h.shape == ref.shape and out.past_key_values is not None
                        and (h - ref).abs().max() <= 0.02 * ref.abs().max())
        except Exception:  # noqa: BLE001
            return False

    @torch.no_grad()
    def run(self, ids, past=None, use_cache=True):
        """Post-norm hidden states [B, T, H] for `ids` and the updated cache."""
        if self.use_base:
            out = self.base(input_ids=ids, past_key_values=past, use_cache=use_cache)
            return out.last_hidden_state, getattr(out, "past_key_values", None)
        out = self.model(input_ids=ids, past_key_values=past, use_cache=use_cache)
        return self._cap, getattr(out, "past_key_values", None)

    @torch.no_grad()
    def generate(self, ids, max_new: int, stop_ids=STOP_IDS, deadline: float | None = None):
        """Greedy decoding for a batch of equal-length prompts (no padding needed).

        Returns generated tokens [B, S], per-row lengths (stop token included), the prompt
        hidden states [B, P, H] and the hidden states of the fed-back tokens [B, S-1, H].
        Rows still running at `deadline` (time.time()) are cut there: a prefix of a greedy
        answer is still valid training data.
        """
        import time
        B = ids.shape[0]
        hp, past = self.run(ids)
        h = hp[:, -1]
        stop = torch.tensor(stop_ids, device=ids.device)
        alive = torch.ones(B, dtype=torch.bool, device=ids.device)
        lens = torch.zeros(B, dtype=torch.long, device=ids.device)
        toks, hs = [], []
        for t in range(max_new):
            nxt = F.linear(h, self.W).float().argmax(-1)
            toks.append(nxt)
            done = alive & torch.isin(nxt, stop)
            lens[done] = t + 1
            alive &= ~done
            if t == max_new - 1 or not bool(alive.any()) or (deadline and time.time() > deadline):
                break
            hstep, past = self.run(nxt[:, None], past)
            h = hstep[:, -1]
            hs.append(h)
        lens[alive] = len(toks)
        steps = torch.stack(hs, 1) if hs else hp[:, :0]
        return torch.stack(toks, 1), lens, hp, steps
