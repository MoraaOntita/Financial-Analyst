# AI AGENT DESIGN (MCP) — FINANCIAL ANALYST ASSISTANT

## 1. CORE GOAL

### Primary Goal:
Help financial analysts explore, understand, and extract insights from used car market data in Supabase using natural language—faster than traditional SQL or dashboards.

### Expanded Goal:
- Reduce time to insight (minutes → seconds)
- Remove dependency on SQL / BI tools
- Provide actionable, explainable insights
- Support decision-making (pricing, valuation, trends)


## 2. AGENT IDENTITY

A conversational financial data analyst that translates natural language into structured insights from used car data.


## 3. DATA UNDERSTANDING CAPABILITIES

- Understand dataset schema:
  car_name, year, selling_price, present_price,
  kms_driven, fuel_type, transmission,
  seller_type, owner

- Identify:
  - Numerical vs categorical fields
  - Relationships (e.g., depreciation over time)

- Infer derived features:
  - Car age = current_year - year
  - Depreciation = present_price - selling_price


## 4. CORE CAPABILITIES

### A. NATURAL LANGUAGE → QUERY
- Convert user questions into structured queries (SQL via MCP tools)
Examples:
  - “Average price of diesel cars”
  - “Cars with lowest depreciation”
  - “Manual vs automatic price difference”

### B. DESCRIPTIVE ANALYTICS
- Compute averages, medians, counts
- Group-by analysis:
  - Price by fuel type
  - Price by year
  - Ownership vs resale value

### C. COMPARATIVE ANALYSIS
- Compare categories:
  - Petrol vs diesel
  - Manual vs automatic
- Return:
  - Direct answer
  - Supporting numbers
  - Short explanation

### D. TREND ANALYSIS
- Identify:
  - Price vs year trends
  - Mileage impact on price
  - Ownership impact

### E. OUTLIER DETECTION
- Detect:
  - Overpriced cars
  - Underpriced cars
  - Unusual mileage-price combinations

### F. PREDICTIVE INSIGHTS
- Estimate fair selling price
- Estimate depreciation rates
- Identify high resale value segments

### G. INVESTMENT INSIGHTS
- Answer:
  - “Which cars retain value best?”
  - “Best segment for resale?”


## 5. CONVERSATIONAL CAPABILITIES (MCP)

- Maintain context across queries
  Example:
    User: “Compare petrol vs diesel”
    User: “What about manual cars?”
    → Agent understands continuation

- Handle clarifications:
  - Ask follow-up questions when needed
  - Resolve ambiguity

- Provide explanations:
  - Not just numbers, but reasoning behind results


## 6. TOOLING (MCP ARCHITECTURE)

- Supabase Query Tool:
  - Execute SQL queries
  - Return structured results

- Aggregation Tool:
  - avg, min, max, count
  - group-by operations

- Statistical Tool (optional):
  - Correlation analysis
  - Simple regression

- Feature Engineering Tool:
  - Compute derived metrics (age, depreciation)


## 7. AGENT WORKFLOW

For each user query:

1. Understand user intent
2. Map to relevant dataset fields
3. Generate structured query
4. Call MCP tool (Supabase)
5. Process results
6. Generate insight
7. Explain in natural language
8. Suggest follow-up insights


## 8. EXAMPLE INTERACTION

User:
“Do diesel cars have better resale value?”

Agent:
- Query average selling_price grouped by fuel_type
- Compare diesel vs petrol
- Respond:

“Diesel cars have higher resale value on average (~X% more than petrol). 
This may be due to better fuel efficiency and higher demand for long-distance use.”


## 9. ADVANCED CAPABILITIES (OPTIONAL)

- Proactive insight generation:
  - Suggest trends or anomalies automatically

- Natural language → charts:
  - Generate visualizations from queries

- Scenario simulation:
  - “What if mileage increases?”

- User modes:
  - Beginner (simple explanations)
  - Analyst (detailed statistics)


## 10. MCP ADVANTAGE

- Tool-based reasoning
- Context-aware conversations
- Combines structured data with natural language understanding
- Enables multi-step analytical workflows