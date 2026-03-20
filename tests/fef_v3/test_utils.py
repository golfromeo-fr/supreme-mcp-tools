#!/usr/bin/env python3
"""
FEF V3 Test Utilities

Helper functions and utilities for testing the Flexible Extensibility Framework V3.
"""
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
            "details": self.details
        }


@dataclass
class ExtensionTestResult:
    """Result of testing an extension."""
    extension_name: str
    availability: TestResult
    execution: Optional[TestResult] = None
    validation: Optional[TestResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "extension_name": self.extension_name,
            "availability": self.availability.to_dict()
        }
        if self.execution:
            result["execution"] = self.execution.to_dict()
        if self.validation:
            result["validation"] = self.validation.to_dict()
        return result


class HTTPClient:
    """HTTP client for making test requests."""
    
    def __init__(self, base_url: str, timeout: int = 10):
        """
        Initialize HTTP client.
        
        Args:
            base_url: Base URL for requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> requests.Response:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.get(url, params=params, timeout=self.timeout)
    
    def post(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            headers: Request headers
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        return self.session.post(url, json=data, headers=headers, timeout=self.timeout)
    
    def put(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> requests.Response:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            headers: Request headers
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        return self.session.put(url, json=data, headers=headers, timeout=self.timeout)
    
    def delete(self, endpoint: str) -> requests.Response:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        return self.session.delete(url, timeout=self.timeout)
    
    def close(self):
        """Close the session."""
        self.session.close()


class ExtensionTester:
    """Tester for FEF V3 extensions."""
    
    def __init__(self, client: HTTPClient):
        """
        Initialize extension tester.
        
        Args:
            client: HTTP client instance
        """
        self.client = client
    
    def test_availability(self, extension_name: str) -> TestResult:
        """
        Test if an extension is available.
        
        Args:
            extension_name: Name of the extension
            
        Returns:
            TestResult
        """
        start_time = time.time()
        
        try:
            response = self.client.get(f"extensions/{extension_name}")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return TestResult(
                    name=f"{extension_name}_availability",
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} is available",
                    details={"response": response.json()}
                )
            else:
                return TestResult(
                    name=f"{extension_name}_availability",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned status {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text}
                )
                
        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_availability",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Timeout accessing {extension_name}"
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_availability",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error accessing {extension_name}: {str(e)}"
            )
    
    def test_execution(
        self,
        extension_name: str,
        params: Dict[str, Any] = None
    ) -> TestResult:
        """
        Test executing an extension.
        
        Args:
            extension_name: Name of the extension
            params: Parameters to pass to the extension
            
        Returns:
            TestResult
        """
        start_time = time.time()
        
        try:
            response = self.client.post(f"extensions/{extension_name}", data=params or {})
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return TestResult(
                        name=f"{extension_name}_execution",
                        status=TestStatus.PASSED,
                        duration_ms=duration_ms,
                        message=f"Extension {extension_name} executed successfully",
                        details={"response": data}
                    )
                except json.JSONDecodeError:
                    return TestResult(
                        name=f"{extension_name}_execution",
                        status=TestStatus.FAILED,
                        duration_ms=duration_ms,
                        message=f"Extension {extension_name} returned invalid JSON",
                        details={"response": response.text}
                    )
            else:
                return TestResult(
                    name=f"{extension_name}_execution",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned status {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text}
                )
                
        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_execution",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Timeout executing {extension_name}"
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_execution",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error executing {extension_name}: {str(e)}"
            )
    
    def test_validation(
        self,
        extension_name: str,
        params: Dict[str, Any],
        expected_fields: List[str]
    ) -> TestResult:
        """
        Test if extension response contains expected fields.
        
        Args:
            extension_name: Name of the extension
            params: Parameters to pass to the extension
            expected_fields: List of expected field names
            
        Returns:
            TestResult
        """
        start_time = time.time()
        
        try:
            response = self.client.post(f"extensions/{extension_name}", data=params)
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return TestResult(
                    name=f"{extension_name}_validation",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned status {response.status_code}",
                    details={"status_code": response.status_code}
                )
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                return TestResult(
                    name=f"{extension_name}_validation",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} returned invalid JSON",
                    details={"response": response.text}
                )
            
            # Check for expected fields
            missing_fields = []
            for field in expected_fields:
                if field not in data:
                    missing_fields.append(field)
            
            if missing_fields:
                return TestResult(
                    name=f"{extension_name}_validation",
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Extension {extension_name} missing fields: {', '.join(missing_fields)}",
                    details={
                        "expected_fields": expected_fields,
                        "missing_fields": missing_fields,
                        "actual_fields": list(data.keys())
                    }
                )
            
            return TestResult(
                name=f"{extension_name}_validation",
                status=TestStatus.PASSED,
                duration_ms=duration_ms,
                message=f"Extension {extension_name} response validated successfully",
                details={
                    "expected_fields": expected_fields,
                    "actual_fields": list(data.keys())
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResult(
                name=f"{extension_name}_validation",
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Error validating {extension_name}: {str(e)}"
            )
    
    def test_extension(
        self,
        extension_name: str,
        params: Dict[str, Any] = None,
        expected_fields: List[str] = None
    ) -> ExtensionTestResult:
        """
        Run all tests for an extension.
        
        Args:
            extension_name: Name of the extension
            params: Parameters to pass to the extension
            expected_fields: List of expected field names for validation
            
        Returns:
            ExtensionTestResult
        """
        availability = self.test_availability(extension_name)
        
        if availability.status != TestStatus.PASSED:
            return ExtensionTestResult(
                extension_name=extension_name,
                availability=availability
            )
        
        execution = self.test_execution(extension_name, params)
        
        if execution.status != TestStatus.PASSED:
            return ExtensionTestResult(
                extension_name=extension_name,
                availability=availability,
                execution=execution
            )
        
        validation = None
        if expected_fields:
            validation = self.test_validation(extension_name, params, expected_fields)
        
        return ExtensionTestResult(
            extension_name=extension_name,
            availability=availability,
            execution=execution,
            validation=validation
        )


class PerformanceTester:
    """Tester for performance metrics."""
    
    def __init__(self, client: HTTPClient):
        """
        Initialize performance tester.
        
        Args:
            client: HTTP client instance
        """
        self.client = client
    
    def test_response_time(
        self,
        endpoint: str,
        method: str = "GET",
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test average response time for an endpoint.
        
        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, PUT, DELETE)
            params: Query parameters
            data: Request body data
            iterations: Number of iterations
            
        Returns:
            Dictionary with performance metrics
        """
        durations = []
        
        for _ in range(iterations):
            start_time = time.time()
            
            try:
                if method.upper() == "GET":
                    self.client.get(endpoint, params)
                elif method.upper() == "POST":
                    self.client.post(endpoint, data)
                elif method.upper() == "PUT":
                    self.client.put(endpoint, data)
                elif method.upper() == "DELETE":
                    self.client.delete(endpoint)
            except:
                pass
            
            durations.append((time.time() - start_time) * 1000)
        
        return {
            "iterations": iterations,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
            "total_ms": sum(durations)
        }
    
    def test_concurrent_requests(
        self,
        endpoint: str,
        method: str = "GET",
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        concurrent: int = 10
    ) -> Dict[str, Any]:
        """
        Test concurrent requests to an endpoint.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body data
            concurrent: Number of concurrent requests
            
        Returns:
            Dictionary with performance metrics
        """
        import concurrent.futures
        
        def make_request():
            start_time = time.time()
            try:
                if method.upper() == "GET":
                    self.client.get(endpoint, params)
                elif method.upper() == "POST":
                    self.client.post(endpoint, data)
                elif method.upper() == "PUT":
                    self.client.put(endpoint, data)
                elif method.upper() == "DELETE":
                    self.client.delete(endpoint)
                return (time.time() - start_time) * 1000, True
            except Exception as e:
                return (time.time() - start_time) * 1000, False
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_duration_ms = (time.time() - start_time) * 1000
        durations = [r[0] for r in results]
        successes = sum(1 for r in results if r[1])
        
        return {
            "concurrent_requests": concurrent,
            "successful_requests": successes,
            "failed_requests": concurrent - successes,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
            "total_duration_ms": total_duration_ms
        }


class ConfigLoader:
    """Loader for test configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load test configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    @staticmethod
    def get_tool_config(config: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific tool.
        
        Args:
            config: Full configuration dictionary
            tool_name: Name of the tool
            
        Returns:
            Tool configuration or None if not found
        """
        return config.get("tools", {}).get(tool_name)
    
    @staticmethod
    def get_enabled_tools(config: Dict[str, Any]) -> List[str]:
        """
        Get list of enabled tools from configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            List of enabled tool names
        """
        tools = config.get("tools", {})
        return [name for name, tool_config in tools.items() 
                if tool_config.get("enabled", True)]


class ReportGenerator:
    """Generator for test reports."""
    
    @staticmethod
    def generate_summary(results: List[ExtensionTestResult]) -> Dict[str, Any]:
        """
        Generate summary from test results.
        
        Args:
            results: List of extension test results
            
        Returns:
            Summary dictionary
        """
        total = len(results)
        passed = sum(1 for r in results 
                   if r.execution and r.execution.status == TestStatus.PASSED)
        failed = sum(1 for r in results 
                   if r.execution and r.execution.status == TestStatus.FAILED)
        errors = sum(1 for r in results 
                   if r.availability.status == TestStatus.ERROR)
        
        return {
            "total_extensions": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": (passed / total * 100) if total > 0 else 0
        }
    
    @staticmethod
    def save_results(results: List[Dict[str, Any]], output_path: str):
        """
        Save test results to JSON file.
        
        Args:
            results: List of test result dictionaries
            output_path: Path to output file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def wait_for_server(
    base_url: str,
    timeout: int = 30,
    interval: float = 0.5
) -> bool:
    """
    Wait for a server to become available.
    
    Args:
        base_url: Base URL of the server
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds
        
    Returns:
        True if server is available, False otherwise
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(base_url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        
        time.sleep(interval)
    
    return False
