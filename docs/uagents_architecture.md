# uAgents Framework - Architecture Documentation

**Version**: 2.0.0

---

## 1. Static Architecture - High-Level Class Diagram

### 1.1 Core Inheritance Hierarchies

The framework is built on three main inheritance trees:

1. **Pydantic BaseModel** → Data validation and configuration
2. **LangChain LLM** → Model abstraction
3. **Python ABC** → Abstract interfaces for extensibility

### 1.2 High-Level Class Diagram


```mermaid
classDiagram
    %% ============================================
    %% EXTERNAL BASE CLASSES (from dependencies)
    %% ============================================
    class BaseModel {
        <<Pydantic>>
        +model_config
        +model_validate()
        +model_dump()
    }

    class LLM {
        <<LangChain>>
        +_call()*
        +_stream()*
        +invoke()
    }

    class ABC {
        <<Python>>
        +abstractmethod
    }

    %% ============================================
    %% PYDANTIC HIERARCHY (Configuration & Entities)
    %% ============================================
    class BaseRunnable {
        +id: str
        +__hash__()
        +_align_lc_tool_to_utool()
    }

    class UAgent {
        +name: str
        +instruction: str
        +llm: ULlm
        +tools: List~UTool~
        +guidelines: InputLLMGuardrail
        +from_dict()$
        +from_json_file()$
    }

    class URunnable {
        +name: str
        +tool: BaseTool
        +setup()
    }

    class UTool {
        +tool_type: str
        +func_name: str
        +description: str
        +args: Dict
        +tool: BaseTool
        +cache_outputs: bool
        +return_directly: bool
    }

    class InferenceParams {
        +temperature: float
        +top_p: float
        +max_tokens: int
        +to_dict()
    }

    %% ============================================
    %% LANGCHAIN LLM HIERARCHY
    %% ============================================
    class ULlm {
        +llm: UModel
        +llm_family: str
        +model_params: InferenceParams
        +_call()
        +_stream()
    }

    %% ============================================
    %% ABC HIERARCHY (Abstract Interfaces)
    %% ============================================
    class UModel {
        <<abstract>>
        +generate()*
    }

    class OpenAIModel {
        +model_name: str
        +base_url: str
        +generate()
    }

    class BedrockModel {
        +model_name: str
        +generate()
    }

    class SagemakerModel {
        +endpoint_name: str
        +generate()
    }

    class Guardrail {
        <<abstract>>
        +gtype: str
        +default_answer: str
        +check()*
    }

    class InputLLMGuardrail {
        +instruction: str
        +check()
    }

    class OutputLLMGuardrail {
        +instruction: str
        +check()
    }

    class InputSafetyGuardrail {
        +check()
    }

    class OutputSafetyGuardrail {
        +check()
    }

    %% ============================================
    %% PLAIN CLASSES (Orchestration & State)
    %% ============================================
    class Driver {
        +flow: Flow
        +memory: AgenticMemory
        +environment: Environment
        +guardrails: List~Guardrail~
        +step_execution_workflow: StepExecutionWorkflow
        +step()
        +step_stream()
    }

    class Flow {
        +conversational: bool
        +agents: Set~AgentType~
        +transitions: Set
        +start_agent: AgentType
        +add_agent()
        +add_transition()
        +add_conditional_transition()
        +build()
    }

    class Environment {
        +functions_module: Any
        +tool_executor: ToolExecutor
        +step()
        +reset()
    }

    class AgenticMemory {
        +session_id: str
        +conversation: Conversation
        +step_memory: StepMemory
        +tools: List~UTool~
        +tool_usage: Dict
    }

    class StepExecutionWorkflow {
        +flow: Flow
        +memory: AgenticMemory
        +environment: Environment
        +app: CompiledStateGraph
        +initialize()
    }

    class TransitionPayload {
        <<enumeration>>
        PASS_INTERMEDIATE_STEPS
        SUMMARY
        LAST_STEP
        NONE
    }

    %% ============================================
    %% INHERITANCE RELATIONSHIPS (Centered)
    %% ============================================

    %% Pydantic Hierarchy
    BaseModel <|-- BaseRunnable
    BaseModel <|-- UTool
    BaseModel <|-- InferenceParams
    BaseRunnable <|-- UAgent
    BaseRunnable <|-- URunnable

    %% LangChain Hierarchy
    LLM <|-- ULlm

    %% ABC Hierarchies
    ABC <|-- UModel
    ABC <|-- Guardrail
    UModel <|-- OpenAIModel
    UModel <|-- BedrockModel
    UModel <|-- SagemakerModel
    Guardrail <|-- InputLLMGuardrail
    Guardrail <|-- OutputLLMGuardrail
    Guardrail <|-- InputSafetyGuardrail
    Guardrail <|-- OutputSafetyGuardrail

    %% ============================================
    %% COMPOSITION & ASSOCIATIONS
    %% ============================================

    %% Driver Composition (owns)
    Driver *-- Flow
    Driver *-- AgenticMemory
    Driver *-- Environment
    Driver *-- StepExecutionWorkflow
    Driver o-- Guardrail

    %% Flow Composition
    Flow o-- UAgent
    Flow o-- URunnable
    Flow ..> TransitionPayload

    %% UAgent Dependencies
    UAgent --> ULlm
    UAgent o-- UTool
    UAgent --> InputLLMGuardrail

    %% ULlm Dependencies
    ULlm --> UModel
    ULlm --> InferenceParams

    %% Environment Dependencies
    Environment ..> AgenticMemory
    Environment ..> UTool

    %% Memory Structure
    AgenticMemory o-- UTool

    %% StepExecutionWorkflow Dependencies
    StepExecutionWorkflow --> Flow
    StepExecutionWorkflow --> AgenticMemory
    StepExecutionWorkflow --> Environment
```

### 1.3 Core Components

#### Pydantic BaseModel Hierarchy
Foundation for all validated configuration objects:

- **BaseRunnable**: Base for executable components (agents, runnables)
- **UAgent**: LLM-powered decision-making agent
- **URunnable**: Deterministic code block wrapper
- **UTool**: Tool specification with validation
- **InferenceParams**: LLM generation parameters

#### LangChain LLM Hierarchy
Integration with LangChain ecosystem:

- **ULlm**: Model-agnostic LLM wrapper extending LangChain's LLM base

#### ABC Hierarchy
Extensible abstract interfaces:

- **UModel**: Abstract base for LLM providers (OpenAI, Bedrock, SageMaker, etc.)
- **Guardrail**: Abstract base for validation rules (Input/Output, LLM/Regex/Safety)

#### Orchestration Layer (Plain Classes)
Runtime execution and state management:

- **Driver**: Central orchestrator, facade for framework
- **Flow**: Multi-agent workflow definition and compilation
- **Environment**: Tool execution context
- **AgenticMemory**: Conversation and execution state
- **StepExecutionWorkflow**: LangGraph state machine executor

---

### 1.4 Key Architectural Patterns

| Pattern | Implementation | Purpose |
|---------|---------------|---------|
| **Template Method** | BaseRunnable → UAgent/URunnable | Standardized agent/runnable lifecycle |
| **Strategy** | UModel implementations | Pluggable LLM providers |
| **Adapter** | ULlm wraps various LLMs | Unified interface for heterogeneous models |
| **Facade** | Driver | Simplifies complex workflow execution |
| **Chain of Responsibility** | Guardrail list processing | Sequential validation pipeline |
| **State Machine** | StepExecutionWorkflow | Agent transition management |
| **Composition over Inheritance** | Driver composition | Flexible runtime assembly |

---

### 1.5 Dependency Flow

```
Consumer
   ↓
Driver (Facade)
   ├→ Flow (contains UAgent/URunnable)
   │    └→ UAgent → ULlm → UModel
   ├→ AgenticMemory (state tracking)
   ├→ Environment (tool execution)
   └→ Guardrails (validation)
```

---

### 1.6 Design Principles

1. **Pydantic-first**: All configuration objects inherit from BaseModel for validation
2. **ABC for extensibility**: Abstract classes define extension points (UModel, Guardrail)
3. **LangChain compatibility**: ULlm extends LangChain's LLM for ecosystem integration
4. **Composition in orchestration**: Driver, Flow, Environment use composition, not inheritance
5. **Clear separation**: Configuration (Pydantic) vs. Runtime (Plain Classes) vs. Extensions (ABC)

---

## 2. Dynamic Architecture - Sequence Diagrams

### 2.1 Single Agent Execution Flow

This diagram shows the complete execution flow for a basic single-agent interaction without tools.

```mermaid
sequenceDiagram
    actor Consumer
    participant Driver
    participant Flow
    participant AgenticMemory
    participant Environment
    participant StepExecutionWorkflow
    participant CompiledStateGraph
    participant agent as UAgent
    participant agent_runnable as agent.agent_runnable
    participant ULlm
    participant UModel
    participant UJSONAgentOutputParser

    %% Initialization Phase
    Consumer->>+Driver: __init__(agent)
    Driver->>+Flow: __init__(conversational=True)
    Flow-->>-Driver: flow
    Driver->>+Flow: add_agent(agent)
    Flow-->>-Driver:
    Driver->>+Flow: build()
    Flow-->>-Driver:
    Driver->>+AgenticMemory: __init__(tools)
    AgenticMemory-->>-Driver: memory
    Driver->>+Environment: __init__()
    Environment-->>-Driver: environment
    Driver->>+StepExecutionWorkflow: __init__(flow, memory, environment)
    StepExecutionWorkflow->>StepExecutionWorkflow: initialize()
    Note over StepExecutionWorkflow: Creates LangGraph CompiledStateGraph
    StepExecutionWorkflow-->>-Driver: workflow
    Driver-->>-Consumer: driver

    %% Execution Phase
    Consumer->>+Driver: step("What is 2+2?")
    Driver->>+AgenticMemory: new_step()
    AgenticMemory-->>-Driver:
    Driver->>+AgenticMemory: step.add_user_input(message)
    AgenticMemory-->>-Driver:

    Note over Driver: Prepare inputs dict with message, context, etc.

    Driver->>+CompiledStateGraph: invoke(inputs, config)
    Note over CompiledStateGraph: LangGraph executes state machine
    CompiledStateGraph->>+StepExecutionWorkflow: _run_agent_lg(data)
    StepExecutionWorkflow->>agent: [retrieve from flow.agents_lookup]
    Note over agent,agent_runnable: UAgent is configuration container<br/>agent_runnable is execution delegate
    StepExecutionWorkflow->>+agent_runnable: invoke(data)

    Note over agent_runnable: RunnableSequence pipeline:
    agent_runnable->>agent_runnable: RunnablePassthrough.assign(agent_scratchpad)
    agent_runnable->>agent_runnable: | ChatPromptTemplate
    agent_runnable->>+ULlm: | llm_with_stop._call(prompt)
    ULlm->>+UModel: generate(prompt, params)
    UModel-->>-ULlm: response_text
    ULlm-->>-agent_runnable: response
    agent_runnable->>+UJSONAgentOutputParser: parse(response)
    UJSONAgentOutputParser-->>-agent_runnable: AgentFinish

    agent_runnable-->>-StepExecutionWorkflow: AgentFinish(output)
    StepExecutionWorkflow->>+AgenticMemory: step.add_intermediate_steps()
    AgenticMemory-->>-StepExecutionWorkflow:
    StepExecutionWorkflow-->>-CompiledStateGraph: {agent_outcome, ...}
    CompiledStateGraph-->>-Driver: outputs

    Note over Driver: Extract answer from outputs["agent_outcome"]

    Driver->>+AgenticMemory: step.add_ai_output(answer)
    AgenticMemory-->>-Driver:
    Driver->>+AgenticMemory: step_memory.add_step(step)
    AgenticMemory-->>-Driver:
    Driver-->>-Consumer: {value, state, intermediate_steps, ...}
```

### 2.2 Agent Execution with Tool Call

This diagram shows the iterative loop when an agent uses tools. The LangGraph state machine loops until the agent returns `AgentFinish`.

```mermaid
sequenceDiagram
    actor Consumer
    participant Driver
    participant CompiledStateGraph
    participant StepExecutionWorkflow
    participant agent as UAgent
    participant agent_runnable as agent.agent_runnable
    participant ULlm
    participant UJSONAgentOutputParser
    participant Environment
    participant ToolExecutor

    Consumer->>Driver: step("What is 5 + 7?")
    Note over Driver: [Initialization steps omitted]

    Driver->>CompiledStateGraph: invoke(inputs, config)

    loop Until AgentFinish Returned
        CompiledStateGraph->>StepExecutionWorkflow: _run_agent_lg(data)
        StepExecutionWorkflow->>agent: [retrieve from flow.agents_lookup]
        Note over agent,agent_runnable: Configuration vs. Execution delegation
        StepExecutionWorkflow->>agent_runnable: invoke(data)
        Note over agent_runnable: agent_scratchpad contains previous tool results
        agent_runnable->>ULlm: _call(prompt)
        ULlm-->>agent_runnable: response
        agent_runnable->>UJSONAgentOutputParser: parse(response)

        alt AgentAction: Tool Required
            UJSONAgentOutputParser-->>agent_runnable: AgentAction(tool, tool_input)
            agent_runnable-->>StepExecutionWorkflow: AgentAction
            StepExecutionWorkflow-->>CompiledStateGraph: {agent_outcome: AgentAction}

            Note over CompiledStateGraph: LangGraph routes to action node
            CompiledStateGraph->>StepExecutionWorkflow: _execute_tools_lg(data)
            StepExecutionWorkflow->>Environment: step(agent_action, memory)
            Environment->>ToolExecutor: __call__(tool, tool_args, args_description)
            ToolExecutor->>ToolExecutor: _convert_args()
            ToolExecutor->>ToolExecutor: _execute(tool_func)
            Note over ToolExecutor: Example: add_numbers(5, 7) → "12"
            ToolExecutor-->>Environment: {status_code: SUCCESS, message}
            Environment-->>StepExecutionWorkflow: tool_result
            StepExecutionWorkflow-->>CompiledStateGraph: {intermediate_steps: [(action, result)]}

            Note over CompiledStateGraph: LangGraph routes back to agent node<br/>(line 169 in lg.py: "return" path)

        else AgentFinish: Final Answer Ready
            UJSONAgentOutputParser-->>agent_runnable: AgentFinish(output)
            agent_runnable-->>StepExecutionWorkflow: AgentFinish
            StepExecutionWorkflow-->>CompiledStateGraph: {agent_outcome: AgentFinish}
            Note over CompiledStateGraph: Exit loop, route to END
        end
    end

    CompiledStateGraph-->>Driver: outputs
    Driver-->>Consumer: {value, intermediate_steps, ...}
```

### 2.3 Key Observations

#### Configuration vs. Execution Separation
The framework employs a two-layer architecture following the **Proxy/Delegate pattern**:

**Configuration Layer (UAgent)**:
- Pydantic BaseModel holding design-time configuration: `name`, `instruction`, `llm`, `tools`, `guidelines`
- Acts as a factory: `setup()` method creates the execution delegate (line 249-267 in agent.py)
- Can be serialized, validated, stored (JSON/dict)
- Lives in `flow.agents_lookup` as a registry of agent configurations

**Execution Layer (agent.agent_runnable)**:
- LangChain `RunnableSequence` created by `create_structured_chat_agent()` (line 273-294 in agent.py)
- Stateless pipeline that transforms inputs to outputs
- Runtime-only construct (cannot be serialized)
- Does the actual work of prompting LLM and parsing responses

**At Runtime** (line 246-255 in lg.py):
```python
agent = self.flow.agents_lookup[data['current_agent']]  # Retrieve config
agent_outcome = agent.agent_runnable.invoke(data)        # Delegate execution
```

This separation enables:
- Clean serialization boundary (config vs. runtime)
- Pydantic validation for configuration
- LangChain ecosystem integration for execution
- Testability (can test config and execution separately)

#### Agent Runnable Pipeline (Line 286-293 in agent.py)
The `agent.agent_runnable` is a LangChain `RunnableSequence` composed of:
1. **RunnablePassthrough.assign(agent_scratchpad)** - Adds previous tool results to context
2. **| ChatPromptTemplate** - Formats the prompt with instructions, tools, context
3. **| llm_with_stop** - Invokes ULlm with stop sequences
4. **| UJSONAgentOutputParser** - Parses output into AgentAction or AgentFinish

#### LangGraph State Machine
The `StepExecutionWorkflow` compiles the Flow into a LangGraph `CompiledStateGraph` which:
- Routes between agent nodes and action (tool execution) nodes
- Manages state transitions based on AgentAction vs AgentFinish
- Loops until AgentFinish is returned

#### LangGraph Iterative Loop
The `CompiledStateGraph` implements a state machine with conditional routing (lines 89-105, 168-180 in lg.py):

**Loop Structure**:
1. **Agent Node** invokes `_run_agent_lg()` → returns `AgentAction` or `AgentFinish`
2. **Conditional Routing** (line 89-96):
   - If `AgentAction` → route to **Action Node**
   - If `AgentFinish` → route to **END** (exit loop)
3. **Action Node** invokes `_execute_tools_lg()` → executes tool
4. **Return Path** (line 169) → routes back to **Agent Node** with tool result in `intermediate_steps`
5. **Loop continues** until agent returns `AgentFinish`

This enables:
- **Multi-tool workflows**: Agent can call multiple tools sequentially
- **Reasoning loops**: Agent can analyze tool results and decide next action
- **Dynamic tool chaining**: Agent determines tool order at runtime based on results

---

## Next Sections

- **Section 3**: Extension Points
- **Section 4**: Deployment Patterns

---

## File Locations

```
src/uagents/
├── entity/
│   ├── base.py          # BaseRunnable
│   ├── agent.py         # UAgent
│   ├── runnables.py     # URunnable
│   └── guardrail.py     # Guardrail hierarchy
├── llm/
│   ├── base.py          # ULlm
│   ├── models.py        # UModel hierarchy
│   └── inference.py     # InferenceParams
├── tools/
│   └── tool.py          # UTool
├── executors/
│   └── drivers.py       # Driver
├── workflows/
│   ├── flows.py         # Flow, TransitionPayload
│   └── lg.py            # StepExecutionWorkflow
├── environments/
│   └── base.py          # Environment
└── memory/
    └── memory.py        # AgenticMemory
```
