"""
Quick smoke test — verifies all OpenEnv methods work correctly.
Run: python test_env.py
"""

from environment import EmailTriageEnv, Action


def test_all_tasks():
    print("🧪 Running Email Triage OpenEnv smoke tests...\n")
    passed = 0
    failed = 0

    for task_id in ["easy", "medium", "hard"]:
        print(f"── Task: {task_id} ──")
        env = EmailTriageEnv(task_id=task_id, seed=42)

        # Test reset()
        obs = env.reset()
        assert obs.task_id == task_id, "task_id mismatch"
        assert obs.subject, "missing subject"
        assert obs.body, "missing body"
        print(f"  ✅ reset() → email_id={obs.email_id}, subject={obs.subject[:40]}...")

        # Test state()
        state = env.state()
        assert state["task_id"] == task_id
        assert state["done"] is False
        print(f"  ✅ state() → {state}")

        # Test step() with correct action (to get high score)
        correct_label = env._current_email["label"]
        action = Action(
            priority=correct_label["priority"],
            category=correct_label["category"],
            route=correct_label["route"],
            summary="Test summary for smoke test verification only.",
        )
        obs2, reward, done, info = env.step(action)
        assert done is True, "should be done after 1 step"
        assert 0.0 <= reward.value <= 1.0, f"reward out of range: {reward.value}"
        assert reward.value >= 0.90, f"perfect action should score >=0.9, got {reward.value}"
        print(f"  ✅ step() → score={reward.value:.4f}, done={done}")
        print(f"     breakdown={reward.breakdown}")

        # Test double-step raises error
        try:
            env.step(action)
            print("  ❌ Should have raised RuntimeError on double-step")
            failed += 1
        except RuntimeError:
            print("  ✅ Double-step correctly raises RuntimeError")
            passed += 1

        # Test invalid action raises ValueError
        try:
            env2 = EmailTriageEnv(task_id=task_id, seed=99)
            env2.reset()
            bad_action = Action(priority="INVALID", category="billing", route="billing_team", summary="x")
            env2.step(bad_action)
            print("  ❌ Should have raised ValueError for invalid priority")
            failed += 1
        except ValueError:
            print("  ✅ Invalid action correctly raises ValueError")
            passed += 1

        # Test reset after done
        obs3 = env.reset()
        assert env._done is False
        print(f"  ✅ reset() after done works → new email_id={obs3.email_id}")

        passed += 3  # reset, state, step checks
        print()

    print(f"{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 All tests passed! Environment is ready for submission.")
    else:
        print("⚠️  Some tests failed. Check above output.")


if __name__ == "__main__":
    test_all_tasks()
