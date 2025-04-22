from openai import OpenAI
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Literal
import json
from lib import AgentContext, PromptContext, ToolContext, ChatContext

MODEL_NAME = "${API_MODEL}"  # OpenAI model name
openai_client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: "${API_KEY_REF}",
  defaultHeaders: {
    ${getHeaderLines().join('\n        ')}
  },
})

class UserMessage(BaseModel):
    role: Literal["user"]
    content: str

class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str

with open('copilot_edit_agent.md', 'r', encoding='utf-8') as file:
    copilot_instructions_edit_agent = file.read()

def get_response(
        messages: List[UserMessage | AssistantMessage],
        workflow_schema: str,
        current_workflow_config: str,
        context: AgentContext | PromptContext | ToolContext | ChatContext | None = None,
        copilot_instructions: str = copilot_instructions_edit_agent
    ) -> str:
    # if context is provided, create a prompt for the context
    if context:
        match context:
            case AgentContext():
                context_prompt = f"""
**NOTE**: The user is currently working on the following agent:
{context.agentName}
"""
            case PromptContext():
                context_prompt = f"""
**NOTE**: The user is currently working on the following prompt:
{context.promptName}
"""
            case ToolContext():
                context_prompt = f"""
**NOTE**: The user is currently working on the following tool:
{context.toolName}
"""
            case ChatContext():
                context_prompt = f"""
**NOTE**: The user has just tested the following chat using the workflow above and has provided feedback / question below this json dump:
```json
{json.dumps(context.messages)}
```
"""
    else:
        context_prompt = ""

    # add the workflow schema to the system prompt
    sys_prompt = copilot_instructions.replace("{workflow_schema}", workflow_schema)

    # add the current workflow config to the last user message
    last_message = messages[-1]
    last_message.content = f"""
Context:
The current workflow config is:
```
{current_workflow_config}
```

{context_prompt}

User: {last_message.content}
"""

    updated_msgs = [{"role": "system", "content": sys_prompt}] + [
        message.model_dump() for message in messages
    ]

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=updated_msgs,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content
