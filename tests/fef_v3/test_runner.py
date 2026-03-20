#!/usr/bin/env python3
"""
FEF V3 Automated Test Runner

Comprehensive test suite for the Flexible Extensibility Framework V3.
Tests all MCP tools with their FEF V3 extensions.
"""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' package not installed. Install with: pip install rich")


def curl_request(method: str, url: str, data: Optional[Dict] = None, timeout: int = 30) -> Tuple[int, Dict]:
    """
    Make HTTP request using curl subprocess.
    
    Returns: (status_code, response_json)
    """
    cmd = ['curl', '-s', '-w', '\n%{http_code}', '-X', method, url, '-H', 'Content-Type: application/json']
    
    if data is not None:
        cmd.extend(['-d', json.dumps(data)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Parse output: last line is HTTP status code, rest is response body
        output = result.stdout.strip().split('\n')
        if len(output) >= 2:
            status_code = int(output[-1])
            response_body = '\n'.join(output[:-1])
        else:
            status_code = 200 if result.returncode == 0 else result.returncode
            response_body = output[0] if output else ''
        
        # Try to parse JSON response
        try:
            response_json = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError:
            response_json = {'raw': response_body}
        
        return status_code, response_json
    except subprocess.TimeoutExpired:
        return 504, {'error': 'Request timed out'}
    except Exception as e:
        return 0, {'error': str(e)}


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolTestResults:
    """Test results for a single tool."""
    tool_name: str
    base_url: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def total_duration_ms(self) -> float:
        """Total test duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def passed_count(self) -> int:
        """Count of passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed_count(self) -> int:
        """Count of failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def error_count(self) -> int:
        """Count of error tests."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)
    
    @property
    def skipped_count(self) -> int:
        """Count of skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)


class FEFTestRunner:
    """Main test runner for FEF V3."""
    
    # Tool configurations
    # Note: Ports confirmed from logs/launcher.log
    TOOLS = {
        "webmcp": {
            "mcp_port": 8001,
            "mgmt_port": 9001,
            "extensions": [
                "request_stats", "cache_stats", "tool_info",
                "cache_config", "api_key", "clear_cache", "reset_counters",
                "search_stats", "fetch_stats", "search_config",
                "search_history", "fetch_cache_hits"
            ]
        },
        "simplemcp": {
            "mcp_port": 8002,
            "mgmt_port": 9012,
            "extensions": [
                "request_stats", "cache_stats", "tool_info",
                "cache_config", "api_key", "clear_cache", "reset_counters",
                "tool_usage", "api_response_times", "timeout_config"
            ]
        },
        "ragmcp": {
            "mcp_port": 8004,
            "mgmt_port": 9014,
            "extensions": [
                "request_stats", "cache_stats", "tool_info",
                "cache_config", "api_key", "clear_cache", "reset_counters",
                "vector_db_stats", "embedding_stats", "collection_stats",
                "collection_config", "embedding_config"
            ]
        },
        "convertermcp": {
            "mcp_port": 8003,
            "mgmt_port": 9013,
            "extensions": [
                "request_stats", "cache_stats", "tool_info",
                "cache_config", "api_key", "clear_cache", "reset_counters",
                "conversion_stats", "format_usage", "conversion_queue",
                "storage_usage", "output_config", "parallel_limit"
            ]
        },
        "oraclemcp": {
            "mcp_port": 8000,
            "mgmt_port": 9010,
            "extensions": [
                "request_stats", "cache_stats", "tool_info",
                "cache_config", "api_key", "clear_cache", "reset_counters",
                "query_stats", "connection_pool", "schema_cache",
                "pool_config"
            ]
        }
    }
    
    def __init__(self, tools: Optional[List[str]] = None, verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            tools: List of tools to test (None = test all)
            verbose: Enable verbose output
        """
        self.tools_to_test = tools or list(self.TOOLS.keys())
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE else None
        self.all_results: List[ToolTestResults] = []
        
        # Filter tools to test
        self.tools_to_test = [t for t in self.tools_to_test if t in self.TOOLS]
    
    def print(self, message: str, style: str = None):
        """Print message with optional styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def test_extension_availability(
        self,
        tool_name: str,
        base_url: str,
        extension_name: str
    ) -> TestResult:
        """
        Test if an extension is available and accessible.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            extension_name: Name of the extension to test
            
        Returns:
            TestResult with test outcome
        """
        start_time = time.time()
        
        try:
            status_code, response_json = curl_request("GET", f"{base_url}/extensions/{extension_name}", timeout=30)
            duration_ms = (time.time() - start_time) * 1000
            
            if status_code == 200:
                return TestResult(
                    name=f"{extension_name}_availability",
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} is available",
                    details={"response": response_json}
                )
            else:
                return TestResult(
                    name=f"{extension_name}_availability",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned status {status_code}",
                    details={"status_code": status_code, "response": response_json}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_availability",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error accessing {extension_name}: {str(e)}"
            )
    
    def test_extension_execution(
        self,
        tool_name: str,
        base_url: str,
        extension_name: str,
        params: Dict[str, Any] = None
    ) -> TestResult:
        """
        Test executing an extension with parameters.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            extension_name: Name of the extension to test
            params: Parameters to pass to the extension
            
        Returns:
            TestResult with test outcome
        """
        start_time = time.time()
        
        try:
            # First get extension info to determine type
            status_code, info = curl_request("GET", f"{base_url}/extensions/{extension_name}", timeout=30)
            
            if status_code != 200:
                return TestResult(
                    name=f"{extension_name}_execution",
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=f"Extension {extension_name} not found",
                    details={"status_code": status_code}
                )
            
            ext_type = info.get("type", "data_source")
            
            # Determine the correct endpoint based on extension type
            endpoint_map = {
                "data_source": "query",
                "mutator": "mutate",
                "action": "execute"
            }
            action = endpoint_map.get(ext_type, "query")
            
            # Call the appropriate endpoint
            status_code, data = curl_request(
                "POST",
                f"{base_url}/extensions/{extension_name}/{action}",
                data={"params": params or {}},
                timeout=60
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if status_code == 200:
                return TestResult(
                    name=f"{extension_name}_execution",
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} ({ext_type}) executed successfully",
                    details={"response": data, "type": ext_type}
                )
            else:
                return TestResult(
                    name=f"{extension_name}_execution",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned status {status_code}",
                    details={"status_code": status_code, "response": data}
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_execution",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error executing {extension_name}: {str(e)}"
            )
    
    def test_list_extensions(self, tool_name: str, base_url: str) -> TestResult:
        """
        Test listing all registered extensions.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            
        Returns:
            TestResult with test outcome
        """
        start_time = time.time()
        
        try:
            status_code, data = curl_request("GET", f"{base_url}/extensions", timeout=30)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if status_code == 200:
                # API returns a list directly, not a dict with "extensions" key
                extensions = data if isinstance(data, list) else data.get("extensions", [])
                expected_count = len(self.TOOLS[tool_name]["extensions"])
                
                if len(extensions) >= expected_count:
                    return TestResult(
                        name="list_extensions",
                        status=TestStatus.PASSED,
                        duration_ms=duration_ms,
                        message=f"Found {len(extensions)} extensions (expected >= {expected_count})",
                        details={"extensions": [e.get("name") for e in extensions]}
                    )
                else:
                    return TestResult(
                        name="list_extensions",
                        status=TestStatus.FAILED,
                        duration_ms=duration_ms,
                        message=f"Found {len(extensions)} extensions (expected >= {expected_count})",
                        details={
                            "expected": expected_count,
                            "found": len(extensions),
                            "extensions": [e.get("name") for e in extensions]
                        }
                    )
            else:
                return TestResult(
                    name="list_extensions",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Failed to list extensions: status {status_code}",
                    details={"status_code": status_code}
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name="list_extensions",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error listing extensions: {str(e)}"
            )
    
    def test_common_extensions(self, tool_name: str, base_url: str) -> List[TestResult]:
        """
        Test common FEF V3 extensions.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            
        Returns:
            List of TestResults
        """
        results = []
        
        # Test data sources
        for ext in ["request_stats", "cache_stats", "tool_info"]:
            results.append(self.test_extension_execution(tool_name, base_url, ext, {}))
        
        # Test mutators
        results.append(self.test_extension_execution(
            tool_name, base_url, "cache_config",
            {"max_size": 2000, "ttl": 600, "enabled": True}
        ))
        
        # Test actions
        results.append(self.test_extension_execution(tool_name, base_url, "reset_counters", {}))
        
        return results
    
    def test_tool_specific_extensions(
        self,
        tool_name: str,
        base_url: str
    ) -> List[TestResult]:
        """
        Test tool-specific extensions.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            
        Returns:
            List of TestResults
        """
        results = []
        
        if tool_name == "webmcp":
            results.append(self.test_extension_execution(tool_name, base_url, "search_stats", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "fetch_stats", {}))
            results.append(self.test_extension_execution(
                tool_name, base_url, "search_config",
                {"max_results": 20, "safe_search": "moderate"}
            ))
            results.append(self.test_extension_execution(
                tool_name, base_url, "search_history",
                {"limit": 5}
            ))
            results.append(self.test_extension_execution(tool_name, base_url, "fetch_cache_hits", {}))
        
        elif tool_name == "simplemcp":
            results.append(self.test_extension_execution(tool_name, base_url, "tool_usage", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "api_response_times", {}))
            results.append(self.test_extension_execution(
                tool_name, base_url, "timeout_config",
                {"default_timeout_ms": 60000, "max_timeout_ms": 300000}
            ))
        
        elif tool_name == "ragmcp":
            results.append(self.test_extension_execution(tool_name, base_url, "vector_db_stats", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "embedding_stats", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "collection_stats", {}))
            results.append(self.test_extension_execution(
                tool_name, base_url, "collection_config",
                {"default_collection": "test", "similarity_threshold": 0.7, "max_results": 10}
            ))
        
        elif tool_name == "convertermcp":
            results.append(self.test_extension_execution(tool_name, base_url, "conversion_stats", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "format_usage", {}))
            results.append(self.test_extension_execution(
                tool_name, base_url, "output_config",
                {"encoding": "utf-8", "max_size_mb": 50}
            ))
        
        elif tool_name == "oraclemcp":
            results.append(self.test_extension_execution(tool_name, base_url, "query_stats", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "connection_pool", {}))
            results.append(self.test_extension_execution(tool_name, base_url, "schema_cache", {}))
            results.append(self.test_extension_execution(
                tool_name, base_url, "pool_config",
                {"max_connections": 10, "min_connections": 2}
            ))
        
        return results
    
    def test_error_handling(self, tool_name: str, base_url: str) -> List[TestResult]:
        """
        Test error handling for invalid requests.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            
        Returns:
            List of TestResults
        """
        results = []
        
        # Test non-existent extension
        start_time = time.time()
        try:
            status_code, response_data = curl_request(
                "POST",
                f"{base_url}/extensions/nonexistent_extension/query",
                data={"params": {}},
                timeout=30
            )
            duration_ms = (time.time() - start_time) * 1000
            
            if status_code in [404, 400]:
                results.append(TestResult(
                    name="error_handling_nonexistent",
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message="Correctly handled non-existent extension",
                    details={"status_code": status_code}
                ))
            else:
                results.append(TestResult(
                    name="error_handling_nonexistent",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Unexpected status code {status_code} for non-existent extension",
                    details={"status_code": status_code}
                ))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            results.append(TestResult(
                name="error_handling_nonexistent",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error testing non-existent extension: {str(e)}"
            ))
        
        # Test invalid parameters
        start_time = time.time()
        try:
            status_code, response_data = curl_request(
                "POST",
                f"{base_url}/extensions/cache_config/mutate",
                data={"params": {"max_size": -1, "ttl": "invalid"}},
                timeout=30
            )
            duration_ms = (time.time() - start_time) * 1000
            
            if status_code in [400, 422]:
                results.append(TestResult(
                    name="error_handling_invalid_params",
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message="Correctly handled invalid parameters",
                    details={"status_code": status_code}
                ))
            else:
                results.append(TestResult(
                    name="error_handling_invalid_params",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Unexpected status code {status_code} for invalid params",
                    details={"status_code": status_code}
                ))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            results.append(TestResult(
                name="error_handling_invalid_params",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error testing invalid parameters: {str(e)}"
            ))
        
        return results
    
    def test_tool(self, tool_name: str) -> ToolTestResults:
        """
        Run all tests for a single tool.
        
        Args:
            tool_name: Name of the tool to test
            
        Returns:
            ToolTestResults with all test results
        """
        tool_config = self.TOOLS[tool_name]
        base_url = f"http://localhost:{tool_config['mgmt_port']}"
        
        self.print(f"\n{'='*60}", style="bold blue")
        self.print(f"Testing {tool_name}", style="bold blue")
        self.print(f"Management Server: {base_url}", style="blue")
        self.print(f"{'='*60}", style="bold blue")
        
        results = ToolTestResults(tool_name=tool_name, base_url=base_url)
        
        # Test 1: List extensions
        self.print("\n[1/5] Testing extension listing...", style="yellow")
        result = self.test_list_extensions(tool_name, base_url)
        results.results.append(result)
        self.print(f"  {result.status.value}: {result.message}", 
                  style="green" if result.status == TestStatus.PASSED else "red")
        
        # Test 2: Common extensions
        self.print("\n[2/5] Testing common extensions...", style="yellow")
        common_results = self.test_common_extensions(tool_name, base_url)
        results.results.extend(common_results)
        for r in common_results:
            status_color = "green" if r.status == TestStatus.PASSED else "red"
            self.print(f"  {r.status.value}: {r.name}", style=status_color)
        
        # Test 3: Tool-specific extensions
        self.print("\n[3/5] Testing tool-specific extensions...", style="yellow")
        specific_results = self.test_tool_specific_extensions(tool_name, base_url)
        results.results.extend(specific_results)
        for r in specific_results:
            status_color = "green" if r.status == TestStatus.PASSED else "red"
            self.print(f"  {r.status.value}: {r.name}", style=status_color)
        
        # Test 4: Error handling
        self.print("\n[4/5] Testing error handling...", style="yellow")
        error_results = self.test_error_handling(tool_name, base_url)
        results.results.extend(error_results)
        for r in error_results:
            status_color = "green" if r.status == TestStatus.PASSED else "red"
            self.print(f"  {r.status.value}: {r.name}", style=status_color)
        
        # Test 5: Performance
        self.print("\n[5/5] Testing performance...", style="yellow")
        perf_result = self.test_performance(tool_name, base_url)
        results.results.append(perf_result)
        self.print(f"  {perf_result.status.value}: {perf_result.message}", 
                  style="green" if perf_result.status == TestStatus.PASSED else "red")
        
        results.end_time = time.time()
        return results
    
    def test_performance(self, tool_name: str, base_url: str) -> TestResult:
        """
        Test performance of extension calls.
        
        Args:
            tool_name: Name of the tool
            base_url: Base URL for the management server
            
        Returns:
            TestResult with performance metrics
        """
        try:
            # Test 10 concurrent requests
            import concurrent.futures
            
            def make_request():
                start = time.time()
                try:
                    curl_request(
                        "POST",
                        f"{base_url}/extensions/request_stats/query",
                        data={"params": {}},
                        timeout=30
                    )
                    return (time.time() - start) * 1000
                except:
                    return (time.time() - start) * 1000
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                durations = list(executor.map(lambda _: make_request(), range(10)))
            
            total_duration_ms = (time.time() - start_time) * 1000
            avg_duration_ms = sum(durations) / len(durations)
            max_duration_ms = max(durations)
            
            # Performance thresholds
            avg_threshold = 100  # ms
            max_threshold = 500  # ms
            
            if avg_duration_ms <= avg_threshold and max_duration_ms <= max_threshold:
                return TestResult(
                    name="performance",
                    status=TestStatus.PASSED,
                    duration_ms=total_duration_ms,
                    message=f"Performance acceptable (avg: {avg_duration_ms:.2f}ms, max: {max_duration_ms:.2f}ms)",
                    details={
                        "avg_duration_ms": avg_duration_ms,
                        "max_duration_ms": max_duration_ms,
                        "total_requests": 10
                    }
                )
            else:
                return TestResult(
                    name="performance",
                    status=TestStatus.FAILED,
                    duration_ms=total_duration_ms,
                    message=f"Performance below threshold (avg: {avg_duration_ms:.2f}ms, max: {max_duration_ms:.2f}ms)",
                    details={
                        "avg_duration_ms": avg_duration_ms,
                        "max_duration_ms": max_duration_ms,
                        "avg_threshold_ms": avg_threshold,
                        "max_threshold_ms": max_threshold
                    }
                )
                
        except Exception as e:
            return TestResult(
                name="performance",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=f"Error testing performance: {str(e)}"
            )
    
    def run_all_tests(self) -> List[ToolTestResults]:
        """
        Run tests for all configured tools.
        
        Returns:
            List of ToolTestResults for all tools
        """
        self.print("\n" + "="*60, style="bold")
        self.print("FEF V3 Automated Test Suite", style="bold")
        self.print("="*60, style="bold")
        self.print(f"\nTesting tools: {', '.join(self.tools_to_test)}", style="blue")
        
        for tool_name in self.tools_to_test:
            try:
                results = self.test_tool(tool_name)
                self.all_results.append(results)
            except Exception as e:
                self.print(f"\nError testing {tool_name}: {str(e)}", style="red")
        
        return self.all_results
    
    def print_summary(self):
        """Print a summary of all test results."""
        self.print("\n" + "="*60, style="bold")
        self.print("Test Summary", style="bold")
        self.print("="*60, style="bold")
        
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_tests = 0
        
        for tool_results in self.all_results:
            total_tests += len(tool_results.results)
            total_passed += tool_results.passed_count
            total_failed += tool_results.failed_count
            total_errors += tool_results.error_count
            
            self.print(f"\n{tool_results.tool_name}:", style="bold")
            self.print(f"  Total: {len(tool_results.results)} tests")
            self.print(f"  Passed: {tool_results.passed_count}", style="green")
            self.print(f"  Failed: {tool_results.failed_count}", 
                      style="red" if tool_results.failed_count > 0 else None)
            self.print(f"  Errors: {tool_results.error_count}", 
                      style="red" if tool_results.error_count > 0 else None)
            self.print(f"  Duration: {tool_results.total_duration_ms:.2f}ms")
        
        self.print(f"\n{'='*60}", style="bold")
        self.print("Overall Results:", style="bold")
        self.print(f"  Total Tests: {total_tests}")
        self.print(f"  Passed: {total_passed}", style="green")
        self.print(f"  Failed: {total_failed}", style="red" if total_failed > 0 else None)
        self.print(f"  Errors: {total_errors}", style="red" if total_errors > 0 else None)
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        self.print(f"  Success Rate: {success_rate:.1f}%", 
                  style="green" if success_rate >= 90 else "yellow" if success_rate >= 70 else "red")
        
        if total_failed == 0 and total_errors == 0:
            self.print(f"\n✓ All tests passed!", style="bold green")
        else:
            self.print(f"\n✗ Some tests failed or had errors", style="bold red")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FEF V3 Automated Test Runner"
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs='+',
        default=None,
        help="Tools to test: webmcp simplemcp ragmcp convertermcp oraclemcp"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse tools argument
    tools = None
    if args.tools:
        # Join all provided tools and split by comma or space
        tools = []
        combined = ' '.join(args.tools)
        for t in combined.replace(',', ' ').split():
            t = t.strip()
            if t in FEFTestRunner.TOOLS:
                tools.append(t)
    
    # Create and run tests
    runner = FEFTestRunner(tools=tools, verbose=args.verbose)
    runner.run_all_tests()
    runner.print_summary()
    
    # Exit with appropriate code
    total_failed = sum(r.failed_count + r.error_count for r in runner.all_results)
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
