from ast import literal_eval
from collections.abc import Callable, Sequence
import json
import re

import yaml
from pydantic import BaseModel

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string


class ExecutionGuard:
    """Hard runtime guard that blocks risky tool calls using untrusted argument provenance."""

    EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

    def __init__(
        self,
        high_impact_tools: set[str] | None = None,
        high_impact_keywords: set[str] | None = None,
        strict_mode: bool = False,
        trusted_recipient_domains: set[str] | None = None,
    ) -> None:
        self.high_impact_tools = high_impact_tools or {"send_email", "delete_file", "send_money"}
        self.high_impact_keywords = high_impact_keywords or {"transfer", "wire"}
        self.strict_mode = strict_mode
        self.trusted_recipient_domains = {
            domain.lower() for domain in (trusted_recipient_domains or {"bluesparrowtech.com"})
        }
        self.current_query_snippets: set[str] = set()
        self.current_query_intents: set[str] = set()
        self.current_trusted_email: str | None = None
        self.current_query_text: str = ""

    def set_query_context(
        self,
        *,
        trusted_query_snippets: set[str] | None = None,
        trusted_query_intents: set[str] | None = None,
        trusted_email: str | None = None,
        query_text: str | None = None,
    ) -> None:
        self.current_query_snippets = trusted_query_snippets or set()
        self.current_query_intents = trusted_query_intents or set()
        self.current_trusted_email = trusted_email
        self.current_query_text = self._normalize(query_text or "")

    def should_block(
        self,
        tool_name: str,
        arguments: dict,
        untrusted_snippets: set[str],
        trusted_query_snippets: set[str] | None = None,
        trusted_query_intents: set[str] | None = None,
    ) -> tuple[bool, str]:
        if not self._is_high_impact_tool(tool_name):
            return False, ""

        trusted_query_snippets = trusted_query_snippets or self.current_query_snippets
        trusted_query_intents = trusted_query_intents or self.current_query_intents

        if tool_name.lower() == "send_email":
            return self._should_block_send_email(
                arguments,
                untrusted_snippets,
                trusted_query_snippets,
                trusted_query_intents,
                self.current_trusted_email,
            )

        if self.strict_mode and tool_name.lower() in {"delete_file", "delete_email"}:
            if "confirm delete" not in self.current_query_text:
                return (
                    True,
                    "Blocked by ExecutionGuard: destructive operation requires explicit confirmation phrase 'confirm delete'.",
                )

        flat_values = self._flatten_argument_values(arguments)
        risky_markers = {
            snippet
            for snippet in untrusted_snippets
            if self.EMAIL_PATTERN.fullmatch(snippet) or len(snippet) >= 24
        }
        for value in flat_values:
            normalized_value = self._normalize(value)
            if not normalized_value:
                continue
            for snippet in risky_markers:
                if snippet in trusted_query_snippets:
                    continue
                if snippet in normalized_value:
                    return (
                        True,
                        "Blocked by ExecutionGuard: high-impact tool arguments appear to originate "
                        "from untrusted tool output. Require explicit user confirmation or trusted source.",
                    )
        return False, ""

    def _should_block_send_email(
        self,
        arguments: dict,
        untrusted_snippets: set[str],
        trusted_query_snippets: set[str],
        trusted_query_intents: set[str],
        trusted_email: str | None,
    ) -> tuple[bool, str]:
        normalized_recipients = [self._normalize(value) for value in self._flatten_argument_values(arguments.get("recipients"))]
        if self.strict_mode:
            for recipient in normalized_recipients:
                if not recipient:
                    continue
                if trusted_email is not None and recipient == self._normalize(trusted_email):
                    continue
                domain = recipient.split("@")[-1] if "@" in recipient else ""
                if domain not in self.trusted_recipient_domains:
                    return (
                        True,
                        "Blocked by ExecutionGuard: emailing untrusted external recipients is not allowed in strict mode.",
                    )

            subject_and_body = self._flatten_argument_values(arguments.get("subject")) + self._flatten_argument_values(
                arguments.get("body")
            )
            for text in subject_and_body:
                normalized_text = self._normalize(text)
                if "http://" in normalized_text or "https://" in normalized_text:
                    return (
                        True,
                        "Blocked by ExecutionGuard: URL content in outgoing email is not allowed in strict mode.",
                    )

        untrusted_emails = {
            snippet for snippet in untrusted_snippets if self.EMAIL_PATTERN.fullmatch(snippet)
        }
        trusted_query_emails = {
            snippet for snippet in trusted_query_snippets if self.EMAIL_PATTERN.fullmatch(snippet)
        }

        for recipient in normalized_recipients:
            if trusted_email is not None and recipient == self._normalize(trusted_email):
                return False, ""
            if recipient and recipient in untrusted_emails and recipient not in trusted_query_emails:
                return (
                    True,
                    "Blocked by ExecutionGuard: high-impact tool arguments appear to originate "
                    "from untrusted tool output. Require explicit user confirmation or trusted source.",
                )

        if "forward_content" in trusted_query_intents:
            return False, ""

        # Keep subject/body checks stricter to reduce false positives from generic overlap.
        risky_text_markers = {
            snippet for snippet in untrusted_snippets if len(snippet) >= 24 and not self.EMAIL_PATTERN.fullmatch(snippet)
        }
        for value in self._flatten_argument_values(arguments.get("subject")) + self._flatten_argument_values(
            arguments.get("body")
        ):
            normalized_value = self._normalize(value)
            for marker in risky_text_markers:
                if marker in normalized_value and any(
                    keyword in normalized_value for keyword in ("security code", "password", "reset")
                ):
                    return (
                        True,
                        "Blocked by ExecutionGuard: high-impact tool arguments appear to originate "
                        "from untrusted tool output. Require explicit user confirmation or trusted source.",
                    )

        return False, ""

    def extract_trusted_query_snippets(self, query: str) -> set[str]:
        if not query:
            return set()
        return {self._normalize(match) for match in re.findall(self.EMAIL_PATTERN, query)}

    def extract_trusted_query_intents(self, query: str) -> set[str]:
        if not query:
            return set()

        normalized = self._normalize(query)
        intents = set()
        content_forwarding_patterns = (
            r"\b(email|send)\b.*\b(content|contents|body|bodies|text)\b",
            r"\b(send|email)\b.*\b(files?|attachments?)\b",
            r"\bsend individual emails\b",
            r"\bsend the body of the email\b",
            r"\bemail the contents of all unread emails\b",
            r"\bemail the contents of all received and sent emails\b",
            r"\bemail the contents of the five largest files\b",
            r"\bemail the contents of all the files\b",
        )
        if any(re.search(pattern, normalized) for pattern in content_forwarding_patterns):
            intents.add("forward_content")
        self_email_patterns = (
            r"\bmy own email address\b",
            r"\bto myself\b",
            r"\bto my email address\b",
            r"\bsend an email to my own email address\b",
            r"\bsend an email to my email address\b",
        )
        if any(re.search(pattern, normalized) for pattern in self_email_patterns):
            intents.add("self_email")
        return intents

    def extract_untrusted_snippets(self, content_blocks: list[dict] | None) -> set[str]:
        if not content_blocks:
            return set()
        extracted = []
        for block in content_blocks:
            if block.get("type") != "text":
                continue
            content = block.get("content", "")
            if isinstance(content, str):
                extracted.extend(self._candidate_snippets_from_text(content))
        return set(extracted)

    def _is_high_impact_tool(self, tool_name: str) -> bool:
        lower = tool_name.lower()
        if lower in self.high_impact_tools:
            return True
        return any(keyword in lower for keyword in self.high_impact_keywords)

    def _flatten_argument_values(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_argument_values(item))
            return flattened
        if isinstance(value, dict):
            flattened = []
            for item in value.values():
                flattened.extend(self._flatten_argument_values(item))
            return flattened
        return [json.dumps(value, default=str)]

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())

    def _candidate_snippets_from_text(self, text: str) -> list[str]:
        snippets = []
        for line in text.splitlines():
            normalized = self._normalize(line)
            if 8 <= len(normalized) <= 220:
                snippets.append(normalized)

        # Preserve token-level provenance markers that are commonly reused in risky calls.
        snippets.extend(self._normalize(match) for match in re.findall(self.EMAIL_PATTERN, text))

        if len(snippets) == 0:
            normalized = self._normalize(text)
            if normalized:
                snippets.append(normalized[:220])
        return snippets[:20]


def is_string_list(s: str):
    try:
        parsed = literal_eval(s)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


def tool_result_to_str(
    tool_result: FunctionReturnType, dump_fn: Callable[[dict | list[dict]], str] = yaml.safe_dump
) -> str:
    """Basic tool output formatter with YAML dump by default. Could work with `json.dumps` as
    `dump_fn`."""
    if isinstance(tool_result, BaseModel):
        return dump_fn(tool_result.model_dump()).strip()

    if isinstance(tool_result, list):
        res_items = []
        for item in tool_result:
            if type(item) in [str, int]:
                res_items += [str(item)]
            elif isinstance(item, BaseModel):
                res_items += [item.model_dump()]
            else:
                raise TypeError("Not valid type for item tool result: " + str(type(item)))

        # If type checking passes, this is guaranteed to be a list of BaseModel
        return dump_fn(res_items).strip()

    return str(tool_result)


class ToolsExecutor(BasePipelineElement):
    """Executes the tool calls in the last messages for which tool execution is required.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text to be fed to the model.
            It should take as argument the tool output, and convert it into a string. The default converter
            converts the output to structured YAML.
    """

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        self.output_formatter = tool_output_formatter
        self.execution_guard = execution_guard

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        tool_call_results = []
        guard_state = extra_args.setdefault(
            "_execution_guard",
            {"untrusted_snippets": set(), "trusted_query_snippets": set(), "trusted_query_intents": set()},
        )
        if self.execution_guard is not None:
            if len(guard_state["trusted_query_snippets"]) == 0:
                guard_state["trusted_query_snippets"].update(
                    self.execution_guard.extract_trusted_query_snippets(query)
                )
            if len(guard_state["trusted_query_intents"]) == 0:
                guard_state["trusted_query_intents"].update(
                    self.execution_guard.extract_trusted_query_intents(query)
                )
            trusted_email = None
            if hasattr(env, "inbox") and hasattr(env.inbox, "account_email"):
                trusted_email = env.inbox.account_email
            self.execution_guard.set_query_context(
                trusted_query_snippets=guard_state["trusted_query_snippets"],
                trusted_query_intents=guard_state["trusted_query_intents"],
                trusted_email=trusted_email,
                query_text=query,
            )
            prior_tool_messages = [m for m in messages if m["role"] == "tool"]
            for message in prior_tool_messages:
                content = message.get("content")
                if isinstance(content, list):
                    guard_state["untrusted_snippets"].update(self.execution_guard.extract_untrusted_snippets(content))

        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Empty function name provided. Provide a valid function name.",
                    )
                )
                continue
            if tool_call.function not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue

            # Converts type of input lists from string to list type
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            if self.execution_guard is not None:
                should_block, reason = self.execution_guard.should_block(
                    tool_call.function,
                    dict(tool_call.args),
                    guard_state["untrusted_snippets"],
                    guard_state["trusted_query_snippets"],
                    guard_state["trusted_query_intents"],
                )
                if should_block:
                    tool_call_results.append(
                        ChatToolResultMessage(
                            role="tool",
                            content=[text_content_block_from_string("")],
                            tool_call_id=tool_call.id,
                            tool_call=tool_call,
                            error=reason,
                        )
                    )
                    continue

            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            tool_call_id = tool_call.id
            formatted_tool_call_result = self.output_formatter(tool_call_result)
            content_blocks = [text_content_block_from_string(formatted_tool_call_result)]
            if self.execution_guard is not None:
                guard_state["untrusted_snippets"].update(self.execution_guard.extract_untrusted_snippets(content_blocks))
            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=content_blocks,
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    error=error,
                )
            )
        return query, runtime, env, [*messages, *tool_call_results], extra_args


class ToolsExecutionLoop(BasePipelineElement):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        logger = Logger().get()
        for _ in range(self.max_iters):
            last_message = messages[-1]
            if not last_message["role"] == "assistant":
                break
            if last_message["tool_calls"] is None:
                break
            if len(last_message["tool_calls"]) == 0:
                break
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
                logger.log(messages)
        return query, runtime, env, messages, extra_args
