from __future__ import annotations

from typing import Dict, List

from .base import BaseSession

sessions: Dict[str, type[BaseSession]] = {}

from .birefnet_general import BiRefNetSessionGeneral

sessions[BiRefNetSessionGeneral.name()] = BiRefNetSessionGeneral

from .birefnet_general_lite import BiRefNetSessionGeneralLite

sessions[BiRefNetSessionGeneralLite.name()] = BiRefNetSessionGeneralLite

from .birefnet_portrait import BiRefNetSessionPortrait

sessions[BiRefNetSessionPortrait.name()] = BiRefNetSessionPortrait

from .birefnet_dis import BiRefNetSessionDIS

sessions[BiRefNetSessionDIS.name()] = BiRefNetSessionDIS

from .birefnet_hrsod import BiRefNetSessionHRSOD

sessions[BiRefNetSessionHRSOD.name()] = BiRefNetSessionHRSOD

from .birefnet_cod import BiRefNetSessionCOD

sessions[BiRefNetSessionCOD.name()] = BiRefNetSessionCOD

from .birefnet_massive import BiRefNetSessionMassive

sessions[BiRefNetSessionMassive.name()] = BiRefNetSessionMassive

from .dis_anime import DisSession

sessions[DisSession.name()] = DisSession

from .dis_custom import DisCustomSession

sessions[DisCustomSession.name()] = DisCustomSession

from .dis_general_use import DisSession as DisSessionGeneralUse

sessions[DisSessionGeneralUse.name()] = DisSessionGeneralUse

from .sam import SamSession

sessions[SamSession.name()] = SamSession

from .silueta import SiluetaSession

sessions[SiluetaSession.name()] = SiluetaSession

from .u2net_cloth_seg import Unet2ClothSession

sessions[Unet2ClothSession.name()] = Unet2ClothSession

from .u2net_custom import U2netCustomSession

sessions[U2netCustomSession.name()] = U2netCustomSession

from .u2net_human_seg import U2netHumanSegSession

sessions[U2netHumanSegSession.name()] = U2netHumanSegSession

from .u2net import U2netSession

sessions[U2netSession.name()] = U2netSession

from .u2netp import U2netpSession

sessions[U2netpSession.name()] = U2netpSession

from .bria_rmbg import BriaRmBgSession

sessions[BriaRmBgSession.name()] = BriaRmBgSession

from .ben_custom import BenCustomSession

sessions[BenCustomSession.name()] = BenCustomSession

from .withoutbg import WithoutBgSession

sessions[WithoutBgSession.name()] = WithoutBgSession

sessions_names = list(sessions.keys())
sessions_class = list(sessions.values())

# Sessions that can be constructed from a model name alone, with no credential.
# Interfaces that build a session on the user's behalf (the web UI, the HTTP
# endpoint) should offer these rather than every registered name.
sessions_names_no_credentials = [
    name for name, cls in sessions.items() if not cls.requires_credentials()
]

# Sessions backed by a downloadable local model file. `rembg d` iterates these,
# so remote backends do not show up as a download that does nothing.
sessions_names_downloadable = [name for name, cls in sessions.items() if cls.is_local()]
