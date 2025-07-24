#!/usr/bin/env python3
"""
Quality gate script to enforce minimum quality standards.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml
from junitparser import JUnitXml, Failure, Error


class QualityGate:
    """Quality gate evaluator for RAG system"""
    
    def __init__(self, config_path: str):
        """Initialize with quality gate configuration"""
        self.config = self._load_config(config_path)
        self.results = {
            "passed": False,
            "total_score": 0.0,
            "checks": {},
            "summary": {}
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load quality gate configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return {
                "thresholds": {
                    "unit_test_pass_rate": 0.95,
                    "integration_test_pass_rate": 0.90,
                    "quality_test_pass_rate": 0.80,
                    "code_coverage": 0.75,
                    "average_retrieval_precision": 0.25,
                    "average_generation_rouge": 0.15,
                    "answer_faithfulness": 0.40
                },
                "weights": {
                    "unit_tests": 0.25,
                    "integration_tests": 0.25,
                    "quality_tests": 0.30,
                    "performance_metrics": 0.20
                },
                "fail_on": {
                    "critical_test_failures": True,
                    "security_vulnerabilities": True,
                    "performance_regression": True
                }
            }
    
    def evaluate_test_results(self, unit_results: List[str], integration_results: List[str], 
                            quality_results: List[str]) -> Dict[str, Any]:
        """Evaluate test results against quality gate"""
        
        # Evaluate unit tests
        unit_check = self._evaluate_unit_tests(unit_results)
        self.results["checks"]["unit_tests"] = unit_check
        
        # Evaluate integration tests
        integration_check = self._evaluate_integration_tests(integration_results)
        self.results["checks"]["integration_tests"] = integration_check
        
        # Evaluate quality tests
        quality_check = self._evaluate_quality_tests(quality_results)
        self.results["checks"]["quality_tests"] = quality_check
        
        # Calculate overall score
        weights = self.config["weights"]
        total_score = (
            unit_check["score"] * weights["unit_tests"] +
            integration_check["score"] * weights["integration_tests"] +
            quality_check["score"] * weights["quality_tests"]
        )
        
        self.results["total_score"] = total_score
        self.results["passed"] = self._determine_pass_fail()
        
        return self.results
    
    def _evaluate_unit_tests(self, result_files: List[str]) -> Dict[str, Any]:
        """Evaluate unit test results"""
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for result_file in result_files:
            if Path(result_file).exists():
                xml = JUnitXml.fromfile(result_file)
                for suite in xml:
                    total_tests += suite.tests
                    total_failures += suite.failures
                    total_errors += suite.errors
        
        if total_tests == 0:
            return {"score": 0.0, "passed": False, "details": {"error": "No unit tests found"}}
        
        pass_rate = (total_tests - total_failures - total_errors) / total_tests
        threshold = self.config["thresholds"]["unit_test_pass_rate"]
        
        return {
            "score": min(pass_rate / threshold, 1.0),
            "passed": pass_rate >= threshold,
            "details": {
                "total_tests": total_tests,
                "failures": total_failures,
                "errors": total_errors,
                "pass_rate": pass_rate,
                "threshold": threshold
            }
        }
    
    def _evaluate_integration_tests(self, result_files: List[str]) -> Dict[str, Any]:
        """Evaluate integration test results"""
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for result_file in result_files:
            if Path(result_file).exists():
                xml = JUnitXml.fromfile(result_file)
                for suite in xml:
                    total_tests += suite.tests
                    total_failures += suite.failures
                    total_errors += suite.errors
        
        if total_tests == 0:
            return {"score": 0.0, "passed": False, "details": {"error": "No integration tests found"}}
        
        pass_rate = (total_tests - total_failures - total_errors) / total_tests
        threshold = self.config["thresholds"]["integration_test_pass_rate"]
        
        return {
            "score": min(pass_rate / threshold, 1.0),
            "passed": pass_rate >= threshold,
            "details": {
                "total_tests": total_tests,
                "failures": total_failures,
                "errors": total_errors,
                "pass_rate": pass_rate,
                "threshold": threshold
            }
        }
    
    def _evaluate_quality_tests(self, result_files: List[str]) -> Dict[str, Any]:
        """Evaluate quality test results"""
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for result_file in result_files:
            if Path(result_file).exists():
                xml = JUnitXml.fromfile(result_file)
                for suite in xml:
                    total_tests += suite.tests
                    total_failures += suite.failures
                    total_errors += suite.errors
        
        if total_tests == 0:
            return {"score": 0.0, "passed": False, "details": {"error": "No quality tests found"}}
        
        pass_rate = (total_tests - total_failures - total_errors) / total_tests
        threshold = self.config["thresholds"]["quality_test_pass_rate"]
        
        return {
            "score": min(pass_rate / threshold, 1.0),
            "passed": pass_rate >= threshold,
            "details": {
                "total_tests": total_tests,
                "failures": total_failures,
                "errors": total_errors,
                "pass_rate": pass_rate,
                "threshold": threshold
            }
        }
    
    def evaluate_quality_metrics(self, quality_report_path: str) -> Dict[str, Any]:
        """Evaluate quality metrics from quality report"""
        if not Path(quality_report_path).exists():
            return {"score": 0.0, "passed": False, "details": {"error": "Quality report not found"}}
        
        try:
            with open(quality_report_path, 'r') as f:
                quality_data = json.load(f)
        except Exception as e:
            return {"score": 0.0, "passed": False, "details": {"error": f"Failed to load quality report: {e}"}}
        
        # Extract key metrics
        retrieval_precision = quality_data.get("average_precision_at_5", 0.0)
        generation_rouge = quality_data.get("average_rouge_l", 0.0)
        answer_faithfulness = quality_data.get("average_faithfulness", 0.0)
        
        # Evaluate against thresholds
        thresholds = self.config["thresholds"]
        precision_check = retrieval_precision >= thresholds["average_retrieval_precision"]
        rouge_check = generation_rouge >= thresholds["average_generation_rouge"]
        faithfulness_check = answer_faithfulness >= thresholds["answer_faithfulness"]
        
        # Calculate composite score
        metrics_passed = sum([precision_check, rouge_check, faithfulness_check])
        metrics_score = metrics_passed / 3.0
        
        return {
            "score": metrics_score,
            "passed": metrics_score >= 0.67,  # At least 2/3 metrics must pass
            "details": {
                "retrieval_precision": retrieval_precision,
                "generation_rouge": generation_rouge,
                "answer_faithfulness": answer_faithfulness,
                "precision_check": precision_check,
                "rouge_check": rouge_check,
                "faithfulness_check": faithfulness_check,
                "thresholds": thresholds
            }
        }
    
    def _determine_pass_fail(self) -> bool:
        """Determine overall pass/fail status"""
        # Check critical failures
        fail_conditions = self.config["fail_on"]
        
        # All test suites must pass at minimum level
        unit_passed = self.results["checks"]["unit_tests"]["passed"]
        integration_passed = self.results["checks"]["integration_tests"]["passed"]
        quality_passed = self.results["checks"]["quality_tests"]["passed"]
        
        if not (unit_passed and integration_passed and quality_passed):
            return False
        
        # Overall score must meet minimum threshold
        min_overall_score = 0.70  # 70% overall score required
        if self.results["total_score"] < min_overall_score:
            return False
        
        return True
    
    def generate_report(self) -> str:
        """Generate human-readable quality gate report"""
        report = []
        report.append("🚦 QUALITY GATE EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall status
        status_emoji = "✅" if self.results["passed"] else "❌"
        status_text = "PASSED" if self.results["passed"] else "FAILED"
        report.append(f"{status_emoji} Overall Status: {status_text}")
        report.append(f"📊 Total Score: {self.results['total_score']:.3f}")
        report.append("")
        
        # Individual check results
        report.append("📋 Individual Check Results:")
        report.append("")
        
        for check_name, check_result in self.results["checks"].items():
            emoji = "✅" if check_result["passed"] else "❌"
            score = check_result["score"]
            report.append(f"{emoji} {check_name.replace('_', ' ').title()}: {score:.3f}")
            
            # Add details
            details = check_result["details"]
            if "total_tests" in details:
                report.append(f"   Tests: {details['total_tests']}, "
                            f"Failures: {details['failures']}, "
                            f"Errors: {details['errors']}")
                report.append(f"   Pass Rate: {details['pass_rate']:.3f} "
                            f"(Threshold: {details['threshold']:.3f})")
            report.append("")
        
        # Recommendations
        if not self.results["passed"]:
            report.append("🔧 RECOMMENDATIONS:")
            report.append("")
            
            for check_name, check_result in self.results["checks"].items():
                if not check_result["passed"]:
                    report.append(f"• Fix failing {check_name.replace('_', ' ')}")
                    
                    details = check_result["details"]
                    if "pass_rate" in details and "threshold" in details:
                        needed_improvement = details["threshold"] - details["pass_rate"]
                        report.append(f"  Need to improve pass rate by {needed_improvement:.3f}")
            report.append("")
        
        # Quality metrics summary
        if "quality_metrics" in self.results["checks"]:
            report.append("📊 QUALITY METRICS SUMMARY:")
            metrics = self.results["checks"]["quality_metrics"]["details"]
            report.append(f"• Retrieval Precision: {metrics.get('retrieval_precision', 0):.3f}")
            report.append(f"• Generation ROUGE-L: {metrics.get('generation_rouge', 0):.3f}")
            report.append(f"• Answer Faithfulness: {metrics.get('answer_faithfulness', 0):.3f}")
            report.append("")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Quality gate evaluation for RAG system")
    parser.add_argument("--unit-results", nargs="+", required=True,
                       help="Paths to unit test result XML files")
    parser.add_argument("--integration-results", nargs="+", required=True,
                       help="Paths to integration test result XML files")
    parser.add_argument("--quality-results", nargs="+", required=True,
                       help="Paths to quality test result XML files")
    parser.add_argument("--quality-report", 
                       help="Path to quality metrics report JSON file")
    parser.add_argument("--config", default=".github/quality-gate-config.yml",
                       help="Path to quality gate configuration file")
    parser.add_argument("--output", 
                       help="Path to save detailed results JSON")
    
    args = parser.parse_args()
    
    # Initialize quality gate
    quality_gate = QualityGate(args.config)
    
    # Evaluate test results
    results = quality_gate.evaluate_test_results(
        args.unit_results,
        args.integration_results,
        args.quality_results
    )
    
    # Evaluate quality metrics if provided
    if args.quality_report:
        quality_metrics_check = quality_gate.evaluate_quality_metrics(args.quality_report)
        results["checks"]["quality_metrics"] = quality_metrics_check
        
        # Update overall score to include quality metrics
        weights = quality_gate.config["weights"]
        if "performance_metrics" in weights:
            results["total_score"] = (
                results["checks"]["unit_tests"]["score"] * weights["unit_tests"] +
                results["checks"]["integration_tests"]["score"] * weights["integration_tests"] +
                results["checks"]["quality_tests"]["score"] * weights["quality_tests"] +
                quality_metrics_check["score"] * weights["performance_metrics"]
            )
            results["passed"] = quality_gate._determine_pass_fail()
    
    # Generate and print report
    report = quality_gate.generate_report()
    print(report)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()