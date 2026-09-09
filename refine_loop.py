
import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import local modules (assuming refine_loop.py is in the judge directory)
try:
    from generator import HTMLGenerator
    from main import HTMLErrorDetector
    from LLM import LLM
    from playwright_html_tester import PlaywrightHTMLTester
except ImportError:
    logging.error("Failed to import local modules. Please ensure this script is in the 'judge' directory.")
    import sys
    sys.exit(1)

class RefinementLoop:
    def __init__(self, 
                 target_model_config: Dict[str, Any],
                 judge_model_config: Dict[str, Any],
                 max_iterations: int = 3,
                 passing_threshold: float = 60.0,
                 output_base_dir: str = "refinement_outputs"):
        
        self.max_iterations = max_iterations
        self.passing_threshold = passing_threshold
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Generator (Target Model)
        self.generator = HTMLGenerator(target_model_config)
        
        # Initialize Judge (Evaluator)
        # Assuming unified model for simplicity, or we can split if needed
        self.judge_llm = LLM(
            model_name=judge_model_config.get("model_name", "auto"),
            api_key=judge_model_config.get("api_key"),
            base_url=judge_model_config.get("base_url")
        )
        
        # VLM for visual check (using same as judge LLM if MLLM)
        self.judge_vlm = self.judge_llm 
        
        self.detector = HTMLErrorDetector(
            llm_model=self.judge_llm, 
            vlm_model=self.judge_vlm,
            rl_mode=False # We want full check for refinement
        )

    def load_failed_case(self, case_dir: str) -> Dict[str, Any]:
        """
        Load necessary data from a failed case directory.
        Expects: final_score_report.json, index.html (optional but good for context)
        """
        case_path = Path(case_dir)
        report_path = case_path / "final_score_report.json"
        html_path = case_path / "index.html"
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found at {report_path}")
            
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
            
        html_content = ""
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        
        # Try to infer the original prompt/task from the report or html content if possible
        # Since we don't have the original prompt stored explicitly in the report shown above,
        # we might need to ask the user to provide it or try to reverse-engineer it from the HTML context
        # OR, we assume the user provides the task description when running this script.
        # For now, let's assume we extract what we can.
        
        return {
            "report": report,
            "html_content": html_content,
            "case_dir": case_dir
        }

    def construct_refinement_prompt(self, task_description: Optional[str], previous_html: str, error_report: Dict[str, Any]) -> str:
        """
        Construct a prompt for the target model to fix the errors.
        """
        
        # Extract key errors
        layer1_errors = error_report.get("第一层_语法结构检测", {}).get("errors", [])
        layer2_errors = error_report.get("第二层_物理数学检测", {}).get("errors", [])
        layer3_errors = error_report.get("第三层_视觉一致性检测", {}).get("errors", [])
        layer4_errors = error_report.get("第四层_运行与交互能力检测", {}).get("errors", [])
        
        error_summary = []
        if layer1_errors:
            error_summary.append("Syntax Errors:\n" + "\n".join([f"- {e.get('description')}" for e in layer1_errors]))
        if layer2_errors:
            error_summary.append("Physics/Math Logic Errors:\n" + "\n".join([f"- {e.get('description')}" for e in layer2_errors]))
        if layer3_errors:
            error_summary.append("Visual Consistency Errors:\n" + "\n".join([f"- {e.get('description')}" for e in layer3_errors]))
        if layer4_errors:
            error_summary.append("Runtime/Interaction Errors:\n" + "\n".join([f"- {e.get('description')}" for e in layer4_errors]))
            
        critique = "\n\n".join(error_summary)
        
        task_context = ""
        if task_description:
            task_context = f"""You previously generated an HTML file for the following task:
"{task_description}"
"""
        else:
            task_context = "You previously generated an HTML file, but it contains errors."

        prompt = f"""
You are an expert web developer. {task_context}

However, an automated evaluation system found the following errors in your code:

{critique}

Here is your previous code:
```html
{previous_html}
```

Please fix these errors and provide the corrected, complete HTML code. 
Ensure the code is self-contained (CSS/JS inside), robust, and fully functional.
Do not output markdown explanations, just the code block.
"""
        return prompt

    async def evaluate_generated_html(self, html_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline on the new HTML.
        """
        # 1. Playwright Test
        pw_output_dir = Path(output_dir) / "screenshots"
        pw_output_dir.mkdir(parents=True, exist_ok=True)
        
        playwright_report = None
        screenshots = []
        
        try:
            tester = PlaywrightHTMLTester(
                html_file_path=str(html_path),
                output_dir=str(pw_output_dir),
                headless=True,
                screenshot_mode='key_frames'
            )
            await asyncio.wait_for(tester.run_complete_test(), timeout=30.0)
            
            pw_report_path = pw_output_dir / 'playwright_test_report.json'
            if pw_report_path.exists():
                with open(pw_report_path, 'r', encoding='utf-8') as f:
                    playwright_report = json.load(f)
                if playwright_report and 'screenshots' in playwright_report:
                    screenshots = playwright_report['screenshots']
        except Exception as e:
            logging.warning(f"Playwright testing failed: {e}")

        # 2. Detector Evaluation
        # Optimize screenshots (take first, middle, last)
        optimized_screenshots = screenshots
        if len(screenshots) > 5:
             import random
             first = screenshots[0]
             last = screenshots[-1]
             middle = random.sample(screenshots[1:-1], 3)
             middle.sort(key=lambda x: x.get('timestamp', 0))
             optimized_screenshots = [first] + middle + [last]

        report = await self.detector.detect_async(
            str(html_path),
            playwright_report=playwright_report,
            screenshots=optimized_screenshots
        )
        
        return report

    async def run_loop(self, task_description: Optional[str], case_dir: str):
        """
        Execute the refinement loop.
        """
        logging.info(f"Starting refinement loop for case: {case_dir}")
        
        current_case_data = self.load_failed_case(case_dir)
        current_html = current_case_data["html_content"]
        current_report = current_case_data["report"]
        
        # Try to infer task description from HTML title if not provided
        if not task_description and current_html:
            import re
            match = re.search(r'<title>(.*?)</title>', current_html, re.IGNORECASE)
            if match:
                inferred_title = match.group(1).strip()
                logging.info(f"Inferred task from HTML title: {inferred_title}")
                task_description = f"Create an educational HTML page about: {inferred_title}"
            else:
                logging.info("No task description provided and could not infer from HTML title. Using generic refinement prompt.")
        
        logging.info(f"Task: {task_description or 'Generic Refinement'}")
        
        # Create a specific directory for this run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_base_dir / f"refine_run_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        for iteration in range(1, self.max_iterations + 1):
            logging.info(f"--- Iteration {iteration}/{self.max_iterations} ---")
            
            # 1. Construct Prompt
            prompt = self.construct_refinement_prompt(task_description, current_html, current_report)
            
            # 2. Generate
            logging.info("Generating corrected HTML...")
            new_html_content = self.generator.generate(
                prompt,
                system_prompt="You are a helpful expert web developer. Fix the errors in the provided HTML code.",
                max_tokens=8192
            )
            
            if not new_html_content:
                logging.error("Generation failed (empty content). Stopping.")
                break
                
            # Save new HTML
            iter_dir = run_dir / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            new_html_path = iter_dir / "index.html"
            with open(new_html_path, "w", encoding="utf-8") as f:
                f.write(new_html_content)
                
            # 3. Evaluate
            logging.info("Evaluating new HTML...")
            new_report = await self.evaluate_generated_html(str(new_html_path), str(iter_dir))
            
            # Save report
            with open(iter_dir / "report.json", "w", encoding="utf-8") as f:
                json.dump(new_report, f, ensure_ascii=False, indent=2)
                
            # 4. Check Score
            overview = new_report.get("检测结果总览", {})
            score = overview.get("综合得分", 0)
            logging.info(f"Iteration {iteration} Score: {score}")
            
            if score >= self.passing_threshold:
                logging.info(f"🎉 Success! Threshold met ({score} >= {self.passing_threshold}).")
                # Save as final
                final_dir = run_dir / "final_success"
                final_dir.mkdir(parents=True, exist_ok=True)
                with open(final_dir / "index.html", "w", encoding="utf-8") as f:
                    f.write(new_html_content)
                with open(final_dir / "report.json", "w", encoding="utf-8") as f:
                    json.dump(new_report, f, ensure_ascii=False, indent=2)
                return True
            
            # Update for next iteration
            current_html = new_html_content
            current_report = new_report
            
        logging.info("❌ Max iterations reached without meeting threshold.")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Refine failed HTML cases iteratively.")
    parser.add_argument("--case_dir", type=str, required=True, help="Directory containing the failed case (must have final_score_report.json)")
    parser.add_argument("--task_desc", type=str, default=None, help="Original task description/prompt (Optional, will infer from HTML title if missing)")
    parser.add_argument("--target_model", type=str, default="gpt-4o", help="Model to use for refinement")
    parser.add_argument("--target_api_key", type=str, default=None)
    parser.add_argument("--target_base_url", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model to use for judging")
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_base_url", type=str, default=None)
    parser.add_argument("--max_iters", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=60.0)
    
    args = parser.parse_args()

    
    # Configs
    target_config = {
        "model_name": args.target_model,
        "api_key": args.target_api_key or os.getenv("TARGET_API_KEY"),
        "base_url": args.target_base_url or os.getenv("TARGET_BASE_URL")
    }
    
    judge_config = {
        "model_name": args.judge_model,
        "api_key": args.judge_api_key or os.getenv("JUDGE_API_KEY"),
        "base_url": args.judge_base_url or os.getenv("JUDGE_BASE_URL")
    }
    
    loop = RefinementLoop(
        target_model_config=target_config,
        judge_model_config=judge_config,
        max_iterations=args.max_iters,
        passing_threshold=args.threshold
    )
    
    await loop.run_loop(args.task_desc, args.case_dir)

if __name__ == "__main__":
    asyncio.run(main())


# python refine_loop.py \
#   --case_dir "你的失败案例目录路径" \
#   --task_desc "原始的任务描述（Prompt）" \
#   --target_model "修复用的模型名称" \
#   --target_api_key "API_KEY" \
#   --target_base_url "BASE_URL"