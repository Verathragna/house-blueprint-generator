from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# ---- Special tokens ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3


class BlueprintTokenizer:
    """
    Parameter and layout tokenizer with explicit PAD/BOS/EOS/SEP and support
    for topology-aware adjacency tokens.
    """

    def __init__(
        self,
        *,
        pos_step: int = 2,
        pos_max: int = 40,
        size_buckets_w: Tuple[int, ...] = (10, 12, 14, 16, 18, 20),
        size_buckets_l: Tuple[int, ...] = (8, 10, 12, 14, 16, 20),
    ) -> None:
        self.pos_step = int(pos_step)
        self.pos_max = int(pos_max)
        self.size_buckets_w = tuple(int(x) for x in size_buckets_w)
        self.size_buckets_l = tuple(int(x) for x in size_buckets_l)
        self.room_token_names: List[str] = [
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
        ]
        self.room_display_names: Dict[str, str] = {
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

        base_tokens = [
            *self.room_token_names,
            "ATTIC",
            "ADA",
            "FIREPLACE",
            "VAULTED",
            "OWNER_SUITE_MAIN",
            "OWNER_SUITE_UPPER",
            "BATH_TUB",
            "BATH_SHOWER",
            "BATH_BOTH",
            "CEILING_H",
            "WINDOW_H",
            "DOOR_H",
            "FOUNDATION_SLAB",
            "FOUNDATION_CRAWL",
            "FOUNDATION_BASEMENT",
            "GARAGE_ATTACHED",
            "GARAGE_DETACHED",
            "STYLE_CRAFTSMAN",
            "STYLE_COLONIAL",
            "STYLE_MODERN",
        ]

        # Add discrete size buckets (W/L) in a fixed order
        for w in self.size_buckets_w:
            base_tokens.append(f"W{w}")
        for l in self.size_buckets_l:
            base_tokens.append(f"L{l}")

        # Map common user-facing terms to internal tokens
        self.term_to_token: Dict[str, str] = {
            "master bedroom": "OWNER_SUITE_MAIN",
            "master suite": "OWNER_SUITE_MAIN",
            "primary bedroom": "OWNER_SUITE_MAIN",
            "primary suite": "OWNER_SUITE_MAIN",
            "owner's suite": "OWNER_SUITE_MAIN",
            "master bath": "BATHROOM",
            "half bath": "BATHROOM",
        }

        self.room_name_aliases: Dict[str, str] = {
            "bed": "BEDROOM",
            "bedroom": "BEDROOM",
            "bedrooms": "BEDROOM",
            "bath": "BATHROOM",
            "bathroom": "BATHROOM",
            "bathrooms": "BATHROOM",
            "full bath": "BATHROOM",
            "half bath": "BATHROOM",
            "kitchen": "KITCHEN",
            "living": "LIVING",
            "living room": "LIVING",
            "great room": "LIVING",
            "dining": "DINING",
            "dining room": "DINING",
            "office": "OFFICE",
            "study": "OFFICE",
            "laundry": "LAUNDRY",
            "laundry room": "LAUNDRY",
            "utility": "LAUNDRY",
            "garage": "GARAGE",
            "closet": "CLOSET",
            "walk-in closet": "CLOSET",
            "bonus": "BONUS",
            "bonus room": "BONUS",
        }
        for canonical, display in self.room_display_names.items():
            key = display.lower()
            self.room_name_aliases.setdefault(key, canonical)
            self.room_name_aliases.setdefault(key.replace(" ", ""), canonical)

        tokens: List[str] = list(base_tokens)

        # Discrete x/y position tokens on a configurable grid from 0-pos_max
        for n in range(0, self.pos_max + self.pos_step, self.pos_step):
            tokens.append(f"X{n}")
            tokens.append(f"Y{n}")
        # Adjacency requirement and edge tokens for every room pair (including same-type)
        adjacency_tokens: List[str] = []
        for i, room_a in enumerate(self.room_token_names):
            for j in range(i, len(self.room_token_names)):
                room_b = self.room_token_names[j]
                adjacency_tokens.append(f"ADJREQ_{room_a}_{room_b}")
                adjacency_tokens.append(f"ADJEDGE_{room_a}_{room_b}")
        tokens.extend(adjacency_tokens)

        self.token_to_id: Dict[str, int] = {
            "<PAD>": PAD_ID,
            "<BOS>": BOS_ID,
            "<EOS>": EOS_ID,
            "<SEP>": SEP_ID,
        }
        self.adj_req_pair_to_id: Dict[Tuple[str, str], int] = {}
        self.adj_edge_pair_to_id: Dict[Tuple[str, str], int] = {}
        self.adj_req_id_to_pair: Dict[int, Tuple[str, str]] = {}
        self.adj_edge_id_to_pair: Dict[int, Tuple[str, str]] = {}

        next_id = max(self.token_to_id.values()) + 1
        for tok in tokens:
            if tok in self.token_to_id:
                continue
            self.token_to_id[tok] = next_id
            if tok.startswith("ADJREQ_"):
                pair = self._parse_adjacency_token(tok)
                self.adj_req_pair_to_id[pair] = next_id
                self.adj_req_id_to_pair[next_id] = pair
            elif tok.startswith("ADJEDGE_"):
                pair = self._parse_adjacency_token(tok)
                self.adj_edge_pair_to_id[pair] = next_id
                self.adj_edge_id_to_pair[next_id] = pair
            next_id += 1

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.room_token_ids: Set[int] = {
            self.token_to_id[name] for name in self.room_token_names
        }
        self.room_token_name_to_id: Dict[str, int] = {
            name: self.token_to_id[name] for name in self.room_token_names
        }
        self.default_room_dims: Dict[str, Tuple[str, str]] = {
            "BEDROOM": ("W12", "L12"),
            "BATHROOM": ("W8", "L8"),
            "KITCHEN": ("W14", "L12"),
            "LIVING": ("W16", "L14"),
            "DINING": ("W12", "L10"),
            "OFFICE": ("W10", "L10"),
            "LAUNDRY": ("W8", "L8"),
            "GARAGE": ("W20", "L20"),
            "CLOSET": ("W10", "L10"),
            "BONUS": ("W14", "L14"),
        }

    # ---- helpers ----
    def get_vocab_size(self) -> int:
        return max(self.token_to_id.values()) + 1

    def is_room_token_id(self, token_id: int) -> bool:
        return token_id in self.room_token_ids

    def room_label_to_token(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        key = (
            label.replace("-", " ")
            .replace("_", " ")
            .strip()
            .lower()
        )
        key = " ".join(key.split())
        if key in self.room_name_aliases:
            return self.room_name_aliases[key]
        if key in self.term_to_token:
            candidate = self.term_to_token[key]
            if candidate in self.room_token_name_to_id:
                return candidate
        if "bed" in key and "bath" not in key:
            return "BEDROOM"
        if "bath" in key:
            return "BATHROOM"
        if "kitchen" in key:
            return "KITCHEN"
        if "living" in key:
            return "LIVING"
        if "dining" in key:
            return "DINING"
        if "laundry" in key or "utility" in key:
            return "LAUNDRY"
        if "garage" in key:
            return "GARAGE"
        if "closet" in key:
            return "CLOSET"
        if "office" in key or "study" in key or "den" in key:
            return "OFFICE"
        if "bonus" in key or "flex" in key or "loft" in key:
            return "BONUS"
        return None

    def normalize_room_tokens(self, labels: List[str]) -> List[str]:
        normalized: List[str] = []
        for label in labels:
            token = self.room_label_to_token(label)
            if token:
                normalized.append(token)
        return normalized

    def adjacency_requirements_from_params(
        self, adjacency: Optional[Dict[str, List[str]]]
    ) -> Dict[int, List[int]]:
        if not adjacency:
            return {}
        result: Dict[int, Set[int]] = {}
        for room_label, neighbors in adjacency.items():
            src_token = self.room_label_to_token(room_label)
            if not src_token:
                logger.debug("Skipping unknown adjacency room label '%s'", room_label)
                continue
            src_id = self.room_token_name_to_id.get(src_token)
            if src_id is None:
                continue
            for neighbor_label in neighbors:
                tgt_token = self.room_label_to_token(neighbor_label)
                if not tgt_token:
                    logger.debug(
                        "Skipping unknown adjacency neighbor '%s' for '%s'",
                        neighbor_label,
                        room_label,
                    )
                    continue
                tgt_id = self.room_token_name_to_id.get(tgt_token)
                if tgt_id is None:
                    continue
                result.setdefault(src_id, set()).add(tgt_id)
                result.setdefault(tgt_id, set()).add(src_id)
        return {k: sorted(v) for k, v in result.items()}

    @staticmethod
    def _pair_key(room_a: str, room_b: str) -> Tuple[str, str]:
        return tuple(sorted((room_a, room_b)))

    @staticmethod
    def _parse_adjacency_token(token_name: str) -> Tuple[str, str]:
        _, left, right = token_name.split("_", 2)
        return BlueprintTokenizer._pair_key(left, right)

    def _bucket_size(self, feet: float, buckets: Tuple[int, ...]) -> str:
        best = min(buckets, key=lambda b: abs(b - feet))
        return f"W{best}"

    def _bucket_pos(
        self, coord: float, prefix: str = "X"
    ) -> str:
        """Quantize an x/y coordinate into a discrete token."""
        c = max(0, min(self.pos_max, int(round(coord / self.pos_step) * self.pos_step)))
        return f"{prefix}{c}"

    # ---- PARAMS ENCODING ----
    def encode_params(self, params: dict) -> List[int]:
        ids = [BOS_ID]
        style = (params.get("houseStyle") or "").lower()
        if "craftsman" in style:
            ids.append(self.token_to_id["STYLE_CRAFTSMAN"])
        elif "colonial" in style:
            ids.append(self.token_to_id["STYLE_COLONIAL"])
        elif "modern" in style:
            ids.append(self.token_to_id["STYLE_MODERN"])

        for _ in range(int(params.get("bedrooms", 0))):
            ids.append(self.token_to_id["BEDROOM"])
        for _ in range(int((params.get("bathrooms") or {}).get("full", 0))):
            ids.append(self.token_to_id["BATHROOM"])
        for _ in range(int(params.get("kitchen", 1))):
            ids.append(self.token_to_id["KITCHEN"])
        for _ in range(int(params.get("livingRooms", 1))):
            ids.append(self.token_to_id["LIVING"])
        for _ in range(int(params.get("diningRooms", 1))):
            ids.append(self.token_to_id["DINING"])
        for _ in range(int(params.get("laundryRooms", 1))):
            ids.append(self.token_to_id["LAUNDRY"])

        if params.get("bonusRoom"):
            ids.append(self.token_to_id["BONUS"])
        if params.get("attic"):
            ids.append(self.token_to_id["ATTIC"])

        ada = params.get("ada") or params.get("adaFeatures")
        if ada:
            ids.append(self.token_to_id["ADA"])
        if params.get("fireplace"):
            ids.append(self.token_to_id["FIREPLACE"])
        if params.get("vaultedCeilings"):
            ids.append(self.token_to_id["VAULTED"])

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

        if params.get("ceilingHeight"):
            ids.append(self.token_to_id["CEILING_H"])
        if params.get("windowHeight"):
            ids.append(self.token_to_id["WINDOW_H"])
        if params.get("doorHeight"):
            ids.append(self.token_to_id["DOOR_H"])

        foundation = (params.get("foundationType") or "").lower()
        if "slab" in foundation:
            ids.append(self.token_to_id["FOUNDATION_SLAB"])
        elif "crawl" in foundation:
            ids.append(self.token_to_id["FOUNDATION_CRAWL"])
        elif "basement" in foundation:
            ids.append(self.token_to_id["FOUNDATION_BASEMENT"])

        garage = params.get("garage") or {}
        if garage:
            if garage.get("attached", True):
                ids.append(self.token_to_id["GARAGE_ATTACHED"])
            else:
                ids.append(self.token_to_id["GARAGE_DETACHED"])
            ids.append(self.token_to_id["GARAGE"])

        adjacency = params.get("adjacency") or {}
        if adjacency:
            seen_pairs: Set[Tuple[str, str]] = set()
            for room_label, neighbors in adjacency.items():
                src_token = self.room_label_to_token(room_label)
                if not src_token:
                    continue
                for neighbor_label in neighbors:
                    tgt_token = self.room_label_to_token(neighbor_label)
                    if not tgt_token:
                        continue
                    pair = self._pair_key(src_token, tgt_token)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    token_id = self.adj_req_pair_to_id.get(pair)
                    if token_id is not None:
                        ids.append(token_id)

        ids.append(SEP_ID)
        return ids

    # ---- LAYOUT ENCODING ----
    def encode_layout(self, layout: dict) -> List[int]:
        ids: List[int] = []
        rooms = (layout or {}).get("layout", {}).get("rooms", [])
        for room in rooms:
            token_name = self.room_label_to_token(room.get("type"))
            if not token_name:
                logger.debug("Skipping unknown room type '%s'", room.get("type"))
                continue
            ids.append(self.token_to_id[token_name])

            w = float(room.get("size", {}).get("width", 12))
            l = float(room.get("size", {}).get("length", 12))
            wtok = self._bucket_size(w, self.size_buckets_w)
            ltok = self._bucket_size(l, self.size_buckets_l).replace("W", "L")
            xtok = self._bucket_pos(room.get("position", {}).get("x", 0), prefix="X")
            ytok = self._bucket_pos(room.get("position", {}).get("y", 0), prefix="Y")

            for tk in (wtok, ltok, xtok, ytok):
                token_id = self.token_to_id.get(tk)
                if token_id is not None:
                    ids.append(token_id)

        adjacency_pairs = self._compute_layout_adjacency_pairs(rooms)
        for pair in sorted(adjacency_pairs):
            token_id = self.adj_edge_pair_to_id.get(pair)
            if token_id is not None:
                ids.append(token_id)

        ids.append(EOS_ID)
        return ids

    def build_training_pair(self, params: dict, layout: dict):
        prefix = self.encode_params(params)  # BOS ... SEP
        tgt = self.encode_layout(layout)  # ... EOS
        x = prefix + tgt[:-1]
        y = [PAD_ID] * len(prefix) + tgt[1:]
        assert len(x) == len(y)
        return x, y

    # ---- LAYOUT DECODING ----
    def decode_layout_tokens(self, token_list: List[int]) -> dict:
        rooms: List[dict] = []
        adjacency_map: Dict[str, Set[str]] = defaultdict(set)

        current_room: Optional[dict] = None
        for tid in token_list:
            if tid in (PAD_ID, BOS_ID, SEP_ID):
                continue
            if tid == EOS_ID:
                break

            tok = self.id_to_token.get(tid, "")
            if self.is_room_token_id(tid):
                if current_room:
                    finalised = self._finalise_room(current_room, allow_defaults=True)
                    if finalised:
                        rooms.append(finalised)
                current_room = {
                    "type_token": tok,
                    "width": None,
                    "length": None,
                    "x": None,
                    "y": None,
                }
                continue

            if tok.startswith("W"):
                if current_room is not None:
                    try:
                        current_room["width"] = int(tok[1:])
                    except ValueError:
                        logger.debug("Failed to parse width token '%s'", tok)
                continue

            if tok.startswith("L"):
                if current_room is not None:
                    try:
                        current_room["length"] = int(tok[1:])
                    except ValueError:
                        logger.debug("Failed to parse length token '%s'", tok)
                continue

            if tok.startswith("X"):
                if current_room is not None:
                    try:
                        current_room["x"] = int(tok[1:])
                    except ValueError:
                        logger.debug("Failed to parse X token '%s'", tok)
                continue

            if tok.startswith("Y"):
                if current_room is not None:
                    try:
                        current_room["y"] = int(tok[1:])
                    except ValueError:
                        logger.debug("Failed to parse Y token '%s'", tok)
                    finalised = self._finalise_room(current_room, allow_defaults=True)
                    if finalised:
                        rooms.append(finalised)
                    current_room = None
                continue

            if tid in self.adj_edge_id_to_pair:
                room_a, room_b = self.adj_edge_id_to_pair[tid]
                display_a = self.room_display_names.get(room_a, room_a.title())
                display_b = self.room_display_names.get(room_b, room_b.title())
                adjacency_map[display_a].add(display_b)
                adjacency_map[display_b].add(display_a)

        if current_room:
            finalised = self._finalise_room(current_room, allow_defaults=True)
            if finalised:
                rooms.append(finalised)

        layout_dict = {"layout": {"rooms": rooms}}
        if adjacency_map:
            layout_dict["adjacency"] = {
                room: sorted(neighbors) for room, neighbors in adjacency_map.items()
            }
        return layout_dict

    def decode_layout_tokens_partial(self, token_list: List[int]) -> dict:
        rooms: List[dict] = []
        adjacency_map: Dict[str, Set[str]] = defaultdict(set)
        current_room: Optional[dict] = None

        for tid in token_list:
            if tid in (PAD_ID, BOS_ID, SEP_ID):
                if tid == SEP_ID:
                    current_room = None
                continue
            if tid == EOS_ID:
                break

            tok = self.id_to_token.get(tid, "")
            if self.is_room_token_id(tid):
                if current_room:
                    finalised = self._finalise_room(current_room, allow_defaults=False)
                    if finalised:
                        rooms.append(finalised)
                        completed_types.append(current_room["type_token"])
                current_room = {
                    "type_token": tok,
                    "width": None,
                    "length": None,
                    "x": None,
                    "y": None,
                }
                continue

            if tok.startswith("W") and current_room is not None:
                current_room["width"] = tok[1:]
                continue

            if tok.startswith("L") and current_room is not None:
                current_room["length"] = tok[1:]
                continue

            if tok.startswith("X") and current_room is not None:
                current_room["x"] = tok[1:]
                continue

            if tok.startswith("Y") and current_room is not None:
                current_room["y"] = tok[1:]
                finalised = self._finalise_room(current_room, allow_defaults=False)
                if finalised:
                    rooms.append(finalised)
                current_room = None
                continue

            if tid in self.adj_edge_id_to_pair and rooms:
                room_a, room_b = self.adj_edge_id_to_pair[tid]
                display_a = self.room_display_names.get(room_a, room_a.title())
                display_b = self.room_display_names.get(room_b, room_b.title())
                if display_a in {room["type"] for room in rooms} and display_b in {
                    room["type"] for room in rooms
                }:
                    adjacency_map[display_a].add(display_b)
                    adjacency_map[display_b].add(display_a)

        if current_room:
            finalised = self._finalise_room(current_room, allow_defaults=False)
            if finalised:
                rooms.append(finalised)

        layout_dict = {"layout": {"rooms": rooms}}
        if adjacency_map:
            layout_dict["adjacency"] = {
                room: sorted(neighbors) for room, neighbors in adjacency_map.items()
            }
        return layout_dict

    def _finalise_room(self, room_state: dict, *, allow_defaults: bool) -> Optional[dict]:
        type_token = room_state.get("type_token") or "BONUS"
        display = self.room_display_names.get(type_token, type_token.title())
        default_w, default_l = self.default_room_dims.get(type_token, ("W12", "L12"))

        def _resolve(value, default_token):
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
            if not allow_defaults:
                return None
            return int(default_token[1:])

        width = _resolve(room_state.get("width"), default_w)
        length = _resolve(room_state.get("length"), default_l)
        if width is None or length is None:
            return None

        def _resolve_coord(value):
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
            if not allow_defaults:
                return None
            return 0

        x = _resolve_coord(room_state.get("x"))
        y = _resolve_coord(room_state.get("y"))
        if x is None or y is None:
            return None

        return {
            "type": display,
            "position": {"x": x, "y": y},
            "size": {"width": width, "length": length},
        }

    def _compute_layout_adjacency_pairs(self, rooms: List[dict]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for idx_a, room_a in enumerate(rooms):
            token_a = self.room_label_to_token(room_a.get("type"))
            if not token_a:
                continue
            for idx_b in range(idx_a + 1, len(rooms)):
                room_b = rooms[idx_b]
                token_b = self.room_label_to_token(room_b.get("type"))
                if not token_b:
                    continue
                if self._rooms_share_wall(room_a, room_b):
                    pairs.add(self._pair_key(token_a, token_b))
        return pairs

    @staticmethod
    def _room_bounds(room: dict) -> Tuple[float, float, float, float]:
        x = float((room.get("position") or {}).get("x", 0))
        y = float((room.get("position") or {}).get("y", 0))
        w = float((room.get("size") or {}).get("width", 0))
        l = float((room.get("size") or {}).get("length", 0))
        return x, y, x + w, y + l

    @classmethod
    def _rooms_share_wall(cls, room_a: dict, room_b: dict, tol: float = 1e-6) -> bool:
        ax1, ay1, ax2, ay2 = cls._room_bounds(room_a)
        bx1, by1, bx2, by2 = cls._room_bounds(room_b)

        vertical_touch = (abs(ax2 - bx1) < tol or abs(bx2 - ax1) < tol) and (
            min(ay2, by2) - max(ay1, by1) > 0
        )
        horizontal_touch = (abs(ay2 - by1) < tol or abs(by2 - ay1) < tol) and (
            min(ax2, bx2) - max(ax1, bx1) > 0
        )
        return vertical_touch or horizontal_touch


__all__ = [
    "BlueprintTokenizer",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "SEP_ID",
]
