"""
Data Retriever v2 (AL/IMEX aware)
----------------------------------
A thin, backward-compatible wrapper around the existing RetrieveData
that augments the exported packs with:
  • rho (AL penalty)
  • s_vec (per-client input dimensions)
  • train_time (training horizon length)

It extends the original implementation from `data_retriever`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import configparser

# Import the base implementation (v1 typo removed; we extend data_retriever)
from data_retriever import RetrieveData as _BaseRetrieveData


class RetrieveData(_BaseRetrieveData):
    """Drop-in replacement that augments packs for AL/IMEX use.

    This class calls the original RetrieveData __init__ and then:
      • reads rho from config if available (with safe fallbacks),
      • computes/attaches s_vec and train_time when inferable,
      • injects the new fields into global_learners_pack and HMatrix_pack.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Prefer config2.ini if present (AL/IMEX overrides)
        try:
            cp2 = configparser.ConfigParser()
            if cp2.read("config2.ini"):
                self.config = cp2
        except Exception:
            pass

        # ---- Read rho from config (robust to different layouts) ----
        self.rho: float = self._extract_rho_from_config(default=getattr(self, "rho", 0.0))

        # ---- Infer s_vec (per-client input dimension) if possible ----
        self.s_vec: Dict[int, int] = self._infer_s_vec()

        # ---- Infer training horizon (train_time) if possible ----
        self.train_time: Optional[int] = self._infer_train_time()

        # ---- Augment packs if they exist on the base retriever ----
        if hasattr(self, "global_learners_pack") and isinstance(self.global_learners_pack, dict):
            self.global_learners_pack["rho"] = self.rho

        if hasattr(self, "HMatrix_pack") and isinstance(self.HMatrix_pack, dict):
            self.HMatrix_pack["rho"] = self.rho
            if self.s_vec:
                self.HMatrix_pack["s_vec"] = self._normalize_index_keys(self.s_vec)
            if self.train_time is not None:
                self.HMatrix_pack["train_time"] = int(self.train_time)

    # -------------------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------------------
    def _extract_rho_from_config(self, default: float = 0.0) -> float:
        """Try multiple places to find rho in the loaded config.
        Priority:
          1) [AL]/[al] rho_init
          2) [GLOBAL_PARAMETERS] rho
          3) attribute self.rho (default provided)
        """
        cfg = getattr(self, "config", None)
        if isinstance(cfg, (configparser.ConfigParser, dict)):
            # ConfigParser: use getfloat if section exists
            if isinstance(cfg, configparser.ConfigParser):
                for sec in ("AL", "al"):
                    if cfg.has_section(sec) and cfg.has_option(sec, "rho_init"):
                        try:
                            return cfg.getfloat(sec, "rho_init")
                        except Exception:
                            pass
                if cfg.has_section("GLOBAL_PARAMETERS") and cfg.has_option("GLOBAL_PARAMETERS", "rho"):
                    try:
                        return cfg.getfloat("GLOBAL_PARAMETERS", "rho")
                    except Exception:
                        pass
            else:
                # dict-like config
                for sec in ("AL", "al"):
                    try:
                        al = cfg.get(sec, {})  # type: ignore[attr-defined]
                        if "rho_init" in al:
                            return float(al["rho_init"])
                    except Exception:
                        pass
                try:
                    gp = cfg.get("GLOBAL_PARAMETERS", {})  # type: ignore[attr-defined]
                    if "rho" in gp:
                        return float(gp["rho"])
                except Exception:
                    pass
        # fallback to provided default
        try:
            return float(default)
        except Exception:
            return 0.0

    def _infer_s_vec(self) -> Dict[int, int]:
        """Infer per-client input dimension s_m.
        Tries, in order:
          • self.input_size_vec (dict-like {m: s_m})
          • self.inputs_train / self.train_inputs (dict of arrays with shape (s_m, T))
          • self.inputs (full series)
        Returns an empty dict if nothing is available.
        """
        # 1) Dedicated vector present
        if hasattr(self, "input_size_vec") and isinstance(self.input_size_vec, dict):
            try:
                return {int(k): int(v) for k, v in self.input_size_vec.items()}
            except Exception:
                pass

        # 2) Check common containers for training inputs
        for name in ("inputs_train", "train_inputs", "u_train"):
            arrs = getattr(self, name, None)
            if isinstance(arrs, dict) and arrs:
                try:
                    return {int(k): int(v.shape[0]) for k, v in arrs.items()}
                except Exception:
                    continue

        # 3) Fallback: full inputs
        for name in ("inputs", "u_full", "inputs_all"):
            arrs = getattr(self, name, None)
            if isinstance(arrs, dict) and arrs:
                try:
                    return {int(k): int(v.shape[0]) for k, v in arrs.items()}
                except Exception:
                    continue

        return {}

    def _infer_train_time(self) -> Optional[int]:
        """Infer training horizon length if possible.
        Tries:
          • self.training_time
          • length from any of: states_train / inputs_train / outputs_train
        Returns None if not available.
        """
        if hasattr(self, "training_time"):
            try:
                return int(getattr(self, "training_time"))
            except Exception:
                pass

        # Heuristic: pull T from first available training array
        candidate_names = (
            "states_train",
            "inputs_train",
            "outputs_train",
            "x_train",
            "u_train",
            "y_train",
        )
        for name in candidate_names:
            dct = getattr(self, name, None)
            if isinstance(dct, dict) and dct:
                try:
                    any_arr = next(iter(dct.values()))
                    # Expected shape (dim, T)
                    if hasattr(any_arr, "shape") and len(any_arr.shape) == 2:
                        return int(any_arr.shape[1])
                except Exception:
                    continue
        return None

    @staticmethod
    def _normalize_index_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
        """Ensure integer keys (1..M) for consistency downstream."""
        out: Dict[int, Any] = {}
        for k, v in d.items():
            try:
                kk = int(k)
            except Exception:
                # if key like 'm3' or similar, extract trailing digits
                ks = str(k)
                digits = "".join(ch for ch in ks if ch.isdigit())
                kk = int(digits) if digits else ks  # type: ignore[assignment]
            out[kk] = v
        return out


__all__ = ["RetrieveData"]
