from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# ---- Special tokens ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3

class BlueprintTokenizer:
    """
    Parameter and layout tokenizer with explicit PAD/BOS/EOS/SEP.
    """
    def __init__(self):
        tokens = [
            "BEDROOM", "BATHROOM", "KITCHEN", "LIVING", "DINING", "OFFICE",
            "LAUNDRY", "GARAGE", "CLOSET", "BONUS", "ATTIC",
            "ADA", "FIREPLACE", "VAULTED",
            "OWNER_SUITE_MAIN", "OWNER_SUITE_UPPER",
            "BATH_TUB", "BATH_SHOWER", "BATH_BOTH",
            "CEILING_H", "WINDOW_H", "DOOR_H",
            "FOUNDATION_SLAB", "FOUNDATION_CRAWL", "FOUNDATION_BASEMENT",
            "GARAGE_ATTACHED", "GARAGE_DETACHED",
            "STYLE_CRAFTSMAN", "STYLE_COLONIAL", "STYLE_MODERN",
            "W10","W12","W14","W16","W18","W20","L8","L10","L12","L14","L16","L20",
        ]
        # Map common user-facing terms to internal tokens
        self.term_to_token = {
            "master bedroom": "OWNER_SUITE_MAIN",
            "master suite": "OWNER_SUITE_MAIN",
            "primary bedroom": "OWNER_SUITE_MAIN",
            "primary suite": "OWNER_SUITE_MAIN",
            "owner's suite": "OWNER_SUITE_MAIN",
            "master bath": "BATHROOM",
            "half bath": "BATHROOM",
        }
        # Discrete x/y position tokens on a 2ft grid from 0-40ft
        for n in range(0, 42, 2):
            tokens.append(f"X{n}")
            tokens.append(f"Y{n}")
        self.token_to_id: Dict[str,int] = {"<PAD>": PAD_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID, "<SEP>": SEP_ID}
        i = max(self.token_to_id.values()) + 1
        for t in tokens:
            self.token_to_id[t] = i; i += 1
        self.id_to_token = {v:k for k,v in self.token_to_id.items()}
        self.default_room_dims = {
            "BEDROOM": ("W12","L12"), "BATHROOM": ("W8","L8"),
            "KITCHEN": ("W14","L12"), "LIVING": ("W16","L14"),
            "DINING": ("W12","L10"), "OFFICE": ("W10","L10"),
            "LAUNDRY": ("W8","L8"), "GARAGE": ("W20","L20"), "CLOSET": ("W10","L10")
        }

    # ---- helpers ----
    def get_vocab_size(self) -> int:
        return max(self.token_to_id.values()) + 1

    def _bucket_size(self, feet: float, buckets=(8,10,12,14,16,20)) -> str:
        best = min(buckets, key=lambda b: abs(b - feet))
        return f"W{best}"

    def _bucket_pos(self, coord: float, step: int = 2, max_val: int = 40, prefix: str = "X") -> str:
        """Quantize an x/y coordinate into a discrete token."""
        c = max(0, min(max_val, int(round(coord / step) * step)))
        return f"{prefix}{c}"

    # ---- PARAMS ENCODING ----
    def encode_params(self, params: dict) -> List[int]:
        ids = [BOS_ID]
        style = (params.get("houseStyle") or "").lower()
        if "craftsman" in style: ids.append(self.token_to_id["STYLE_CRAFTSMAN"])
        elif "colonial" in style: ids.append(self.token_to_id["STYLE_COLONIAL"])
        elif "modern" in style: ids.append(self.token_to_id["STYLE_MODERN"])

        for _ in range(int(params.get("bedrooms", 0))): ids.append(self.token_to_id["BEDROOM"])
        for _ in range(int((params.get("bathrooms") or {}).get("full", 0))): ids.append(self.token_to_id["BATHROOM"])
        ids += [self.token_to_id["KITCHEN"], self.token_to_id["LIVING"], self.token_to_id["DINING"], self.token_to_id["LAUNDRY"]]

        if params.get("bonusRoom"): ids.append(self.token_to_id["BONUS"])
        if params.get("attic"): ids.append(self.token_to_id["ATTIC"])

        ada = params.get("ada") or params.get("adaFeatures")
        if ada: ids.append(self.token_to_id["ADA"])
        if params.get("fireplace"): ids.append(self.token_to_id["FIREPLACE"])
        if params.get("vaultedCeilings"): ids.append(self.token_to_id["VAULTED"])

        loc = (params.get("ownerSuiteLocation") or "").lower()
        mapped = self.term_to_token.get(loc)
        if mapped:
            ids.append(self.token_to_id[mapped])
        elif "main" in loc:
            ids.append(self.token_to_id["OWNER_SUITE_MAIN"])
        elif "upper" in loc:
            ids.append(self.token_to_id["OWNER_SUITE_UPPER"])

        bath = (params.get("masterBathOption") or "").lower()
        mapped = self.term_to_token.get(bath)
        if mapped:
            ids.append(self.token_to_id[mapped])
        elif "both" in bath:
            ids.append(self.token_to_id["BATH_BOTH"])
        elif "tub" in bath:
            ids.append(self.token_to_id["BATH_TUB"])
        elif "shower" in bath:
            ids.append(self.token_to_id["BATH_SHOWER"])

        if params.get("ceilingHeight"): ids.append(self.token_to_id["CEILING_H"])
        if params.get("windowHeight"): ids.append(self.token_to_id["WINDOW_H"])
        if params.get("doorHeight"): ids.append(self.token_to_id["DOOR_H"])

        fnd = (params.get("foundationType") or params.get("foundation") or "").lower()
        if "slab" in fnd: ids.append(self.token_to_id["FOUNDATION_SLAB"])
        elif "crawl" in fnd: ids.append(self.token_to_id["FOUNDATION_CRAWL"])
        elif "basement" in fnd: ids.append(self.token_to_id["FOUNDATION_BASEMENT"])

        garage = params.get("garage") or {}
        if garage.get("attached", True): ids.append(self.token_to_id["GARAGE_ATTACHED"])
        else: ids.append(self.token_to_id["GARAGE_DETACHED"])
        ids.append(self.token_to_id["GARAGE"])

        ids.append(SEP_ID)
        return ids

    def encode_layout(self, layout: dict) -> List[int]:
        ids: List[int] = []
        rooms = (layout or {}).get("layout", {}).get("rooms", [])
        for r in rooms:
            t = (r.get("type") or "").lower()
            mapped = self.term_to_token.get(t)
            if mapped:
                ids.append(self.token_to_id[mapped])
            elif "bed" in t: ids.append(self.token_to_id["BEDROOM"])
            elif "bath" in t: ids.append(self.token_to_id["BATHROOM"])
            elif "kitchen" in t: ids.append(self.token_to_id["KITCHEN"])
            elif "living" in t: ids.append(self.token_to_id["LIVING"])
            elif "dining" in t: ids.append(self.token_to_id["DINING"])
            elif "office" in t: ids.append(self.token_to_id["OFFICE"])
            elif "laundry" in t: ids.append(self.token_to_id["LAUNDRY"])
            elif "garage" in t: ids.append(self.token_to_id["GARAGE"])
            elif "closet" in t: ids.append(self.token_to_id["CLOSET"])
            elif "bonus" in t: ids.append(self.token_to_id["BONUS"])

            w = float(r.get("size", {}).get("width", 12))
            l = float(r.get("size", {}).get("length", 12))
            wtok = self._bucket_size(w)  # Wxx
            ltok = self._bucket_size(l).replace("W","L")
            xtok = self._bucket_pos(r.get("position", {}).get("x", 0), prefix="X")
            ytok = self._bucket_pos(r.get("position", {}).get("y", 0), prefix="Y")
            for tk in (wtok, ltok, xtok, ytok):
                if tk in self.token_to_id:
                    ids.append(self.token_to_id[tk])
        ids.append(EOS_ID)
        return ids

    def build_training_pair(self, params: dict, layout: dict):
        prefix = self.encode_params(params)     # BOS ... SEP
        tgt = self.encode_layout(layout)        # ... EOS
        x = prefix + tgt[:-1]
        y = [PAD_ID]*len(prefix) + tgt[1:]
        assert len(x) == len(y)
        return x, y

    def decode_layout_tokens(self, token_list: List[int]) -> dict:
        rooms = []
        last_w = last_l = last_x = last_y = None
        for tid in token_list:
            if tid in (PAD_ID, BOS_ID, SEP_ID):
                continue
            if tid == EOS_ID:
                break
            tok = self.id_to_token.get(tid, "")
            if tok.startswith("W"):
                try:
                    last_w = int(tok[1:])
                except (ValueError, TypeError):
                    logger.debug("Failed to parse width token '%s'", tok)
                continue
            if tok.startswith("L"):
                try:
                    last_l = int(tok[1:])
                except (ValueError, TypeError):
                    logger.debug("Failed to parse length token '%s'", tok)
                continue
            if tok.startswith("X"):
                try:
                    last_x = int(tok[1:])
                except (ValueError, TypeError):
                    logger.debug("Failed to parse X position token '%s'", tok)
                continue
            if tok.startswith("Y"):
                try:
                    last_y = int(tok[1:])
                except (ValueError, TypeError):
                    logger.debug("Failed to parse Y position token '%s'", tok)
                continue
            if tok in (
                "BEDROOM",
                "BATHROOM",
                "KITCHEN",
                "LIVING",
                "DINING",
                "OFFICE",
                "LAUNDRY",
                "GARAGE",
                "CLOSET",
                "BONUS",
            ):
                rtype_map = {
                    "BEDROOM": "Bedroom",
                    "BATHROOM": "Bathroom",
                    "KITCHEN": "Kitchen",
                    "LIVING": "Living Room",
                    "DINING": "Dining Room",
                    "OFFICE": "Office",
                    "LAUNDRY": "Laundry Room",
                    "GARAGE": "Garage",
                    "CLOSET": "Closet",
                    "BONUS": "Bonus Room",
                }
                wtok, ltok = self.default_room_dims.get(tok, ("W12", "L12"))
                width = last_w or int(wtok[1:])
                length = last_l or int(ltok[1:])
                x = last_x or 0
                y = last_y or 0
                rooms.append(
                    {
                        "type": rtype_map[tok],
                        "position": {"x": x, "y": y},
                        "size": {"width": width, "length": length},
                    }
                )
                last_w = last_l = last_x = last_y = None
        return {"layout": {"rooms": rooms}}
