from __future__ import annotations

from typing import List

from .base import BaseSession

sessions_class: List[type[BaseSession]] = []
sessions_names: List[str] = []

from .birefnet_general import BiRefNetSessionGeneral

sessions_class.append(BiRefNetSessionGeneral)
sessions_names.append(BiRefNetSessionGeneral.name())

from .birefnet_general_lite import BiRefNetSessionGeneralLite

sessions_class.append(BiRefNetSessionGeneralLite)
sessions_names.append(BiRefNetSessionGeneralLite.name())

from .birefnet_portrait import BiRefNetSessionPortrait

sessions_class.append(BiRefNetSessionPortrait)
sessions_names.append(BiRefNetSessionPortrait.name())

from .birefnet_dis import BiRefNetSessionDIS

sessions_class.append(BiRefNetSessionDIS)
sessions_names.append(BiRefNetSessionDIS.name())

from .birefnet_hrsod import BiRefNetSessionHRSOD

sessions_class.append(BiRefNetSessionHRSOD)
sessions_names.append(BiRefNetSessionHRSOD.name())

from .birefnet_cod import BiRefNetSessionCOD

sessions_class.append(BiRefNetSessionCOD)
sessions_names.append(BiRefNetSessionCOD.name())

from .birefnet_massive import BiRefNetSessionMassive

sessions_class.append(BiRefNetSessionMassive)
sessions_names.append(BiRefNetSessionMassive.name())

from .dis_anime import DisSession

sessions_class.append(DisSession)
sessions_names.append(DisSession.name())

from .dis_general_use import DisSession as DisSessionGeneralUse

sessions_class.append(DisSessionGeneralUse)
sessions_names.append(DisSessionGeneralUse.name())

from .sam import SamSession

sessions_class.append(SamSession)
sessions_names.append(SamSession.name())

from .silueta import SiluetaSession

sessions_class.append(SiluetaSession)
sessions_names.append(SiluetaSession.name())

from .u2net_cloth_seg import Unet2ClothSession

sessions_class.append(Unet2ClothSession)
sessions_names.append(Unet2ClothSession.name())

from .u2net_custom import U2netCustomSession

sessions_class.append(U2netCustomSession)
sessions_names.append(U2netCustomSession.name())

from .u2net_human_seg import U2netHumanSegSession

sessions_class.append(U2netHumanSegSession)
sessions_names.append(U2netHumanSegSession.name())

from .u2net import U2netSession

sessions_class.append(U2netSession)
sessions_names.append(U2netSession.name())

from .u2netp import U2netpSession

sessions_class.append(U2netpSession)
sessions_names.append(U2netpSession.name())

from .bria_rmbg import BriaRmBgSession

sessions_class.append(BriaRmBgSession)
sessions_names.append(BriaRmBgSession.name())
