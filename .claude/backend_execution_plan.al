# Backend Development Master Plan
# AgentLang program coordinating all backend development phases

# Input: Development plan from project root
project_analysis = breakdown:rootcause(development_plan) → .md
phase_approaches = breakdown:parallel(project_analysis) → .json
phase_evaluation = evaluate:vote(phase_approaches) → .json
best_approach = select:filter(phase_evaluation, phase_approaches) → .md
technical_standards = act:draft(best_approach) → .md
integration_architecture = breakdown:tree(technical_standards) → .md
execution_roadmap = act:draft(integration_architecture) → .md
success_metrics = breakdown:tree(execution_roadmap) → .md
master_plan_implementation = act:implement(success_metrics)
master_plan_test = evaluate:test(master_plan_implementation) → .json

END PROGRAM