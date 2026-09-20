"""Installed dimOS module and external blueprint for Forepaw.

The module is intentionally thin: dimOS owns tool discovery, MCP transport,
capability arbitration, and process lifecycle; the loopback controller service
owns the simulator and is the only process allowed to publish motion.
"""

from __future__ import annotations

from dimos.agents.annotation import skill
from dimos.agents.mcp.mcp_server import McpServer
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.module import Module, ModuleConfig

from .controller_service import HttpControllerClient
from .skills import WorldModelSkillFacade


class ForepawConfig(ModuleConfig):
    controller_url: str = "http://127.0.0.1:8787"
    request_timeout_s: float = 120.0


class ForepawSkills(Module):
    """dimOS skill surface for action-conditioned Go2 predictive control."""

    config: ForepawConfig
    dedicated_worker = True

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        client = HttpControllerClient(
            self.config.controller_url,
            timeout_s=self.config.request_timeout_s,
        )
        # Fail during module construction when the sole motion owner is absent.
        client.health()
        self._facade = WorldModelSkillFacade(client)

    @skill
    def imagine(self, candidate_commands: list[list[dict]]) -> str:
        """Predict six-block futures for candidate commands without robot motion."""

        return self._facade.imagine(candidate_commands)

    @skill(uses=["movement"])
    def plan_to(self, x: float, y: float) -> str:
        """Plan, lock, execute block zero, observe, and replan toward (x, y)."""

        return self._facade.plan_to(x, y)

    @skill
    def stop_motion(self) -> str:
        """Preempt an active Forepaw run and confirm its zero-velocity block."""

        return self._facade.stop()


forepaw_blueprint = autoconnect(
    ForepawSkills.blueprint(),
    McpServer.blueprint(),
).global_config(viewer="none")

__all__ = ["ForepawConfig", "ForepawSkills", "forepaw_blueprint"]
