from tests.utils import BaseTest
from bean.core import Scheduler
import time
from threading import Event

class TestScheduler(BaseTest):

    def test_task_runs_multiple_times(self):
        counter = { "n": 0 }

        def fn():
            counter["n"] += 1

        Scheduler().task(fn, runs=3).start().join()

        self.assertEqual(counter["n"], 3)

    def test_job_runs_periodically(self):
        counter = { "n": 0 }

        def fn():
            counter["n"] += 1

        Scheduler().job(fn, interval=0.01, runs=5).start().join()

        self.assertEqual(counter["n"], 5)

    def test_delay(self):
        event = Event()

        def fn():
            event.set()

        s = Scheduler().task(fn, delay=0.5).start()

        # Should NOT run immediately
        self.assertFalse(event.is_set())

        s.join()

        self.assertTrue(event.is_set())

    def test_stop_prevents_future_runs(self):
        counter = { "n": 0 }

        def fn():
            counter["n"] += 1

        s = Scheduler().job(fn, interval=0.5).start()
        time.sleep(0.3)
        s.stop()

        current = counter["n"]
        time.sleep(0.1)

        # Should not increase after stop
        self.assertEqual(counter["n"], current)

    def test_exception_does_not_crash_scheduler(self):
        counter = { "n": 0 }

        def fn():
            counter["n"] += 1
            if counter["n"] == 1:
                raise RuntimeError("boom")

        s = Scheduler().job(fn, interval=0.01, runs=3).start()
        time.sleep(0.1)
        s.join()

        # Should still complete remaining runs
        self.assertEqual(counter["n"], 3)

