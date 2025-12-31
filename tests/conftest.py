import os
import pytest

SLOW_TEST_WARNING_SECONDS = float(os.getenv("SLOW_TEST_WARNING_SECONDS", "5"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    if SLOW_TEST_WARNING_SECONDS <= 0:
        return
    if report.duration < SLOW_TEST_WARNING_SECONDS:
        return

    message = (
        "WARNING: slow test "
        f"{report.nodeid} took {report.duration:.2f}s "
        f"(threshold {SLOW_TEST_WARNING_SECONDS:.2f}s)"
    )
    reporter = item.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(message)
    else:
        print(message)
