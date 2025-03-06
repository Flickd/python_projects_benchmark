# Copyright 2009-2025 Joshua Bronson. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


#                             * Code review nav *
#                        (see comments in __init__.py)
# ============================================================================
# ← Prev: _abc.py              Current: _base.py            Next: _frozen.py →
# ============================================================================


"""Provide :class:`BidictBase`."""

from __future__ import annotations

import typing as t
import weakref
from collections.abc import ItemsView
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Reversible
from collections.abc import ValuesView
from operator import eq
from types import MappingProxyType

from ._abc import BidirectionalMapping
from ._dup import DROP_NEW
from ._dup import DROP_OLD
from ._dup import ON_DUP_DEFAULT
from ._dup import RAISE
from ._dup import OnDup
from ._exc import DuplicationError
from ._exc import KeyAndValueDuplicationError
from ._exc import KeyDuplicationError
from ._exc import ValueDuplicationError
from ._iter import inverted
from ._iter import iteritems
from ._typing import KT
from ._typing import MISSING
from ._typing import OKT
from ._typing import OVT
from ._typing import VT
from ._typing import Maplike
from ._typing import MapOrItems


OldKV = tuple[OKT[KT], OVT[VT]]
DedupResult = t.Optional[OldKV[KT, VT]]
Unwrites = list[tuple[t.Any, ...]]
BT = t.TypeVar('BT', bound='BidictBase[t.Any, t.Any]')


class BidictKeysView(KeysView[KT], ValuesView[KT]):
    """Since the keys of a bidict are the values of its inverse (and vice versa),
    the :class:`~collections.abc.ValuesView` result of calling *bi.values()*
    is also a :class:`~collections.abc.KeysView` of *bi.inverse*.
    """


class BidictBase(BidirectionalMapping[KT, VT]):
    on_dup = ON_DUP_DEFAULT

    _fwdm: MutableMapping[KT, VT]  #: the backing forward mapping (*key* → *val*)
    _invm: MutableMapping[VT, KT]  #: the backing inverse mapping (*val* → *key*)

    # Use Any rather than KT/VT in the following to avoid "ClassVar cannot contain type variables" errors:
    _fwdm_cls: t.ClassVar[type[MutableMapping[t.Any, t.Any]]] = dict  #: class of the backing forward mapping
    _invm_cls: t.ClassVar[type[MutableMapping[t.Any, t.Any]]] = dict  #: class of the backing inverse mapping

    #: The class of the inverse bidict instance.
    _inv_cls: t.ClassVar[type[BidictBase[t.Any, t.Any]]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._init_class()

    @classmethod
    def _init_class(cls) -> None:
        cls._ensure_inv_cls()
        cls._set_reversed()

    __reversed__: t.ClassVar[t.Any]

    @classmethod
    def _set_reversed(cls) -> None:
        if cls is not BidictBase:
            resolved = cls.__reversed__
            overridden = resolved is not BidictBase.__reversed__
            if overridden:  # E.g. OrderedBidictBase, OrderedBidict
                return
        backing_reversible = all(issubclass(i, Reversible) for i in (cls._fwdm_cls, cls._invm_cls))
        cls.__reversed__ = _fwdm_reversed if backing_reversible else None

    @classmethod
    def _ensure_inv_cls(cls) -> None:
        if getattr(cls, '__dict__', {}).get('_inv_cls'):  # Don't assume cls.__dict__ (e.g. mypyc native class)
            return
        cls._inv_cls = cls._make_inv_cls()

    @classmethod
    def _make_inv_cls(cls: type[BT]) -> type[BT]:
        diff = cls._inv_cls_dict_diff()
        cls_is_own_inv = all(getattr(cls, k, MISSING) == v for (k, v) in diff.items())
        if cls_is_own_inv:
            return cls
        # Suppress auto-calculation of _inv_cls's _inv_cls since we know it already.
        # Works with the guard in BidictBase._ensure_inv_cls() to prevent infinite recursion.
        diff['_inv_cls'] = cls
        inv_cls = type(f'{cls.__name__}Inv', (cls, GeneratedBidictInverse), diff)
        inv_cls.__module__ = cls.__module__
        return t.cast(type[BT], inv_cls)

    @classmethod
    def _inv_cls_dict_diff(cls) -> dict[str, t.Any]:
        return {
            '_fwdm_cls': cls._invm_cls,
            '_invm_cls': cls._fwdm_cls,
        }

    def __init__(self, arg: MapOrItems[KT, VT] = (), /, **kw: VT) -> None:
        self._fwdm = self._fwdm_cls()
        self._invm = self._invm_cls()
        self._update(arg, kw, rollback=False)

    @property
    def inverse(self) -> BidictBase[VT, KT]:

        # First check if a strong reference is already stored.
        inv: BidictBase[VT, KT] | None = getattr(self, '_inv', None)
        if inv is not None:
            return inv
        # Next check if a weak reference is already stored.
        invweak = getattr(self, '_invweak', None)
        if invweak is not None:
            inv = invweak()  # Try to resolve a strong reference and return it.
            if inv is not None:
                return inv
        # No luck. Compute the inverse reference and store it for subsequent use.
        inv = self._make_inverse()
        self._inv: BidictBase[VT, KT] | None = inv
        self._invweak: weakref.ReferenceType[BidictBase[VT, KT]] | None = None
        # Also store a weak reference back to `instance` on its inverse instance, so that
        # the second `.inverse` access in `bi.inverse.inverse` hits the cached weakref.
        inv._inv = None
        inv._invweak = weakref.ref(self)
        # In e.g. `bidict().inverse.inverse`, this design ensures that a strong reference
        # back to the original instance is retained before its refcount drops to zero,
        # avoiding an unintended potential deallocation.
        return inv

    def _make_inverse(self) -> BidictBase[VT, KT]:
        inv: BidictBase[VT, KT] = self._inv_cls()
        inv._fwdm = self._invm
        inv._invm = self._fwdm
        return inv

    @property
    def inv(self) -> BidictBase[VT, KT]:
        return self.inverse

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        items = dict(self.items()) if self else ''
        return f'{clsname}({items})'

    def values(self) -> BidictKeysView[VT]:
        return t.cast(BidictKeysView[VT], self.inverse.keys())

    def keys(self) -> KeysView[KT]:
        fwdm, fwdm_cls = self._fwdm, self._fwdm_cls
        return fwdm.keys() if fwdm_cls is dict else BidictKeysView(self)

    def items(self) -> ItemsView[KT, VT]:
        return self._fwdm.items() if self._fwdm_cls is dict else super().items()

    def __contains__(self, key: t.Any) -> bool:
        return key in self._fwdm

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._fwdm.items() == other.items()
        # Ref: https://docs.python.org/3/library/constants.html#NotImplemented
        return NotImplemented

    def equals_order_sensitive(self, other: object) -> bool:
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        return all(map(eq, self.items(), other.items()))

    def _dedup(self, key: KT, val: VT, on_dup: OnDup) -> DedupResult[KT, VT]:
        fwdm, invm = self._fwdm, self._invm
        oldval: OVT[VT] = fwdm.get(key, MISSING)
        oldkey: OKT[KT] = invm.get(val, MISSING)
        isdupkey, isdupval = oldval is not MISSING, oldkey is not MISSING
        if isdupkey and isdupval:
            if key == oldkey:
                assert val == oldval
                # (key, val) duplicates an existing item -> no-op.
                return None
            # key and val each duplicate a different existing item.
            if on_dup.val is RAISE:
                raise KeyAndValueDuplicationError(key, val)
            if on_dup.val is DROP_NEW:
                return None
            assert on_dup.val is DROP_OLD
            # Fall through to the return statement on the last line.
        elif isdupkey:
            if on_dup.key is RAISE:
                raise KeyDuplicationError(key)
            if on_dup.key is DROP_NEW:
                return None
            assert on_dup.key is DROP_OLD
            # Fall through to the return statement on the last line.
        elif isdupval:
            if on_dup.val is RAISE:
                raise ValueDuplicationError(val)
            if on_dup.val is DROP_NEW:
                return None
            assert on_dup.val is DROP_OLD
            # Fall through to the return statement on the last line.
        # else neither isdupkey nor isdupval.
        return oldkey, oldval

    def _write(self, newkey: KT, newval: VT, oldkey: OKT[KT], oldval: OVT[VT], unwrites: Unwrites | None) -> None:
        fwdm, invm = self._fwdm, self._invm
        fwdm_set, invm_set = fwdm.__setitem__, invm.__setitem__
        fwdm_del, invm_del = fwdm.__delitem__, invm.__delitem__
        # Always perform the following writes regardless of duplication.
        fwdm_set(newkey, newval)
        invm_set(newval, newkey)
        if oldval is MISSING and oldkey is MISSING:  # no key or value duplication
            # {0: 1, 2: 3} | {4: 5} => {0: 1, 2: 3, 4: 5}
            if unwrites is not None:
                unwrites.extend((
                    (fwdm_del, newkey),
                    (invm_del, newval),
                ))
        elif oldval is not MISSING and oldkey is not MISSING:  # key and value duplication across two different items
            # {0: 1, 2: 3} | {0: 3} => {0: 3}
            fwdm_del(oldkey)
            invm_del(oldval)
            if unwrites is not None:
                unwrites.extend((
                    (fwdm_set, newkey, oldval),
                    (invm_set, oldval, newkey),
                    (fwdm_set, oldkey, newval),
                    (invm_set, newval, oldkey),
                ))
        elif oldval is not MISSING:  # just key duplication
            # {0: 1, 2: 3} | {2: 4} => {0: 1, 2: 4}
            invm_del(oldval)
            if unwrites is not None:
                unwrites.extend((
                    (fwdm_set, newkey, oldval),
                    (invm_set, oldval, newkey),
                    (invm_del, newval),
                ))
        else:
            assert oldkey is not MISSING  # just value duplication
            # {0: 1, 2: 3} | {4: 3} => {0: 1, 4: 3}
            fwdm_del(oldkey)
            if unwrites is not None:
                unwrites.extend((
                    (fwdm_set, oldkey, newval),
                    (invm_set, newval, oldkey),
                    (fwdm_del, newkey),
                ))

    def _update(
        self,
        arg: MapOrItems[KT, VT],
        kw: Mapping[str, VT] = MappingProxyType({}),
        *,
        rollback: bool | None = None,
        on_dup: OnDup | None = None,
    ) -> None:
        if not isinstance(arg, (Iterable, Maplike)):
            raise TypeError(f"'{arg.__class__.__name__}' object is not iterable")
        if not arg and not kw:
            return
        if on_dup is None:
            on_dup = self.on_dup
        if rollback is None:
            rollback = RAISE in on_dup

        # Fast path when we're empty and updating only from another bidict (i.e. no dup vals in new items).
        if not self and not kw and isinstance(arg, BidictBase):
            self._init_from(arg)
            return

        # Fast path when we're adding more items than we contain already and rollback is enabled:
        # Update a copy of self with rollback disabled. Fail if that fails, otherwise become the copy.
        if rollback and isinstance(arg, t.Sized) and len(arg) + len(kw) > len(self):
            tmp = self.copy()
            tmp._update(arg, kw, rollback=False, on_dup=on_dup)
            self._init_from(tmp)
            return
        
        write = self._write
        unwrites: Unwrites | None = [] if rollback else None
        for key, val in iteritems(arg, **kw):
            try:
                dedup_result = self._dedup(key, val, on_dup)
            except DuplicationError:
                if unwrites is not None:
                    for fn, *args in reversed(unwrites):
                        fn(*args)
                raise
            if dedup_result is not None:
                write(key, val, *dedup_result, unwrites=unwrites)

    def __copy__(self: BT) -> BT:
        """Used for the copy protocol. See the :mod:`copy` module."""
        return self.copy()

    def copy(self: BT) -> BT:
        return self._from_other(self.__class__, self)

    @staticmethod
    def _from_other(bt: type[BT], other: MapOrItems[KT, VT], inv: bool = False) -> BT:
        inst = bt()
        inst._init_from(other)
        return t.cast(BT, inst.inverse) if inv else inst

    def _init_from(self, other: MapOrItems[KT, VT]) -> None:
        self._fwdm.clear()
        self._invm.clear()
        self._fwdm.update(other)
        # If other is a bidict, use its existing backing inverse mapping, otherwise
        # other could be a generator that's now exhausted, so invert self._fwdm on the fly.
        inv = other.inverse if isinstance(other, BidictBase) else inverted(self._fwdm)
        self._invm.update(inv)

    def __or__(self: BT, other: Mapping[KT, VT]) -> BT:
        if not isinstance(other, Mapping):
            return NotImplemented
        new = self.copy()
        new._update(other, rollback=False)
        return new

    def __ror__(self: BT, other: Mapping[KT, VT]) -> BT:
        if not isinstance(other, Mapping):
            return NotImplemented
        new = self.__class__(other)
        new._update(self, rollback=False)
        return new

    def __len__(self) -> int:
        return len(self._fwdm)

    def __iter__(self) -> Iterator[KT]:
        return iter(self._fwdm)

    def __getitem__(self, key: KT) -> VT:
        return self._fwdm[key]

    def __reduce__(self) -> tuple[t.Any, ...]:
        cls = self.__class__
        inst: Mapping[t.Any, t.Any] = self
        if should_invert := isinstance(self, GeneratedBidictInverse):
            cls = self._inv_cls
            inst = self.inverse
        return self._from_other, (cls, dict(inst), should_invert)


# See BidictBase._set_reversed() above.
def _fwdm_reversed(self: BidictBase[KT, t.Any]) -> Iterator[KT]:
    assert isinstance(self._fwdm, Reversible)
    return reversed(self._fwdm)


BidictBase._init_class()


class GeneratedBidictInverse:
    """Base class for dynamically-generated inverse bidict classes."""


#                             * Code review nav *
# ============================================================================
# ← Prev: _abc.py              Current: _base.py            Next: _frozen.py →
# ============================================================================
