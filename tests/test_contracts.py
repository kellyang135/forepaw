from __future__ import annotations

import unittest

from go2wm.contracts import (
    ActionCommand,
    BlockEvents,
    BlockTransition,
    ContactSample,
    ObjectState,
    PhysicsSample,
    Pose2D,
    RGBObservation,
    StateLabels,
)


class ContractTests(unittest.TestCase):
    def test_action_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            ActionCommand(float("nan"), 0.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            ActionCommand(0.0, 0.0, 0.0)

    def test_observation_requires_exact_packed_rgb_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 12 RGB bytes"):
            RGBObservation("f0", "e0", 0.0, "overhead", 2, 2, b"short")

    def test_event_aggregation_preserves_brief_contact_and_fall(self) -> None:
        samples = (
            PhysicsSample(0.01),
            PhysicsSample(0.02, (ContactSample("light", 0.2),)),
            PhysicsSample(0.03),
            PhysicsSample(0.04, (), True),
            PhysicsSample(0.05, out_of_bounds=True),
        )
        events = BlockEvents.aggregate(samples)
        self.assertEqual(events.contacted_object_ids, ("light",))
        self.assertEqual(events.contact_sample_count, 1)
        self.assertEqual(events.first_contact_time_s, 0.02)
        self.assertEqual(events.last_contact_time_s, 0.02)
        self.assertEqual(events.max_normal_impulse_ns, 0.2)
        self.assertTrue(events.fell)
        self.assertEqual(events.fall_sample_count, 1)
        self.assertEqual(events.first_fall_time_s, 0.04)
        self.assertTrue(events.out_of_bounds)
        self.assertEqual(events.out_of_bounds_sample_count, 1)
        self.assertEqual(events.first_out_of_bounds_time_s, 0.05)

    def test_block_rejects_misaligned_high_frequency_samples(self) -> None:
        start = RGBObservation("f0", "e0", 0.0, "cam", 1, 1, b"\x00\x00\x00")
        end = RGBObservation("f1", "e0", 0.5, "cam", 1, 1, b"\x01\x01\x01")
        objects = (ObjectState("box", "red", Pose2D(0.0, 0.0), True),)
        start_labels = StateLabels(0.0, Pose2D(0.0, 0.0), objects)
        end_labels = StateLabels(0.5, Pose2D(0.1, 0.0), objects)
        samples = (PhysicsSample(0.25), PhysicsSample(0.49))
        with self.assertRaisesRegex(ValueError, "block end"):
            BlockTransition(
                start,
                end,
                ActionCommand(0.2, 0.0),
                start_labels,
                end_labels,
                samples,
                BlockEvents.aggregate(samples),
                ActionCommand(0.2, 0.0),
                0.25,
            )


if __name__ == "__main__":
    unittest.main()
