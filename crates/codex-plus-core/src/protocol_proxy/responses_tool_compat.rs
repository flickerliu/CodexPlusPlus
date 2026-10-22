use super::{
    generic_custom_proxy_tool, normalize_responses_item_id, responses_reasoning_text, strip_sse_field,
};
use anyhow::Context;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub struct ResponsesToolCompat {
    custom_tools: BTreeSet<(String, String)>,
}

impl ResponsesToolCompat {
    pub fn prepare_request(request: &mut Value) -> anyhow::Result<Option<Self>> {
        let mut context = Self::default();
        if let Some(tools) = request.get_mut("tools") {
            context.prepare_tools(tools, "")?;
        }
        if let Some(choice) = request.get_mut("tool_choice") {
            Self::prepare_tool_choice(choice);
        }
        if let Some(input) = request.get_mut("input") {
            match input {
                Value::Array(items) => {
                    for item in items {
                        Self::prepare_history_item(item)?;
                    }
                }
                Value::Object(_) => Self::prepare_history_item(input)?,
                _ => {}
            }
        }
        Ok((!context.custom_tools.is_empty()).then_some(context))
    }

    fn prepare_tools(&mut self, tools: &mut Value, namespace: &str) -> anyhow::Result<()> {
        for tool in tools.as_array_mut().context("Responses tools 必须是数组")? {
            match tool.get("type").and_then(Value::as_str) {
                Some("namespace") => {
                    let name = tool
                        .get("name")
                        .and_then(Value::as_str)
                        .context("Responses namespace 缺少名称")?
                        .to_string();
                    let children = tool
                        .get_mut("tools")
                        .context("Responses namespace 缺少 tools")?;
                    self.prepare_tools(children, &name)?;
                }
                Some("custom") => {
                    let name = tool
                        .get("name")
                        .and_then(Value::as_str)
                        .filter(|name| !name.is_empty())
                        .context("Responses custom 工具缺少名称")?
                        .to_string();
                    anyhow::ensure!(
                        self.custom_tools
                            .insert((namespace.to_string(), name.clone())),
                        "Responses custom 工具名称重复：{name}"
                    );
                    let description = tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let wrapped = generic_custom_proxy_tool(&name, description);
                    let mut description = wrapped["function"]["description"]
                        .as_str()
                        .context("custom 工具转换缺少描述")?
                        .to_string();
                    description.push_str(
                        "\nSupply the original freeform text in the input string argument.",
                    );
                    if let Some(format) = tool.get("format") {
                        description.push_str(
                            "\nThe input string must follow this original tool format:\n",
                        );
                        description.push_str(&serde_json::to_string(format)?);
                    }
                    let object = tool
                        .as_object_mut()
                        .context("Responses custom 工具必须是对象")?;
                    object.remove("format");
                    object.insert("type".to_string(), json!("function"));
                    object.insert("description".to_string(), json!(description));
                    object.insert(
                        "parameters".to_string(),
                        wrapped["function"]["parameters"].clone(),
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn prepare_tool_choice(choice: &mut Value) {
        if choice.get("type").and_then(Value::as_str) == Some("custom") {
            choice["type"] = json!("function");
        }
        if let Some(tools) = choice.get_mut("tools").and_then(Value::as_array_mut) {
            for tool in tools {
                Self::prepare_tool_choice(tool);
            }
        }
    }

    fn prepare_history_item(item: &mut Value) -> anyhow::Result<()> {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning")
                if item.get("id").and_then(Value::as_str)
                    .and_then(|id| id.strip_prefix("rs_resp_"))
                    .is_some_and(|suffix| !suffix.is_empty())
                    && !item.get("encrypted_content")
                        .is_some_and(|value| !value.is_null() && value.as_str() != Some("")) =>
            {
                let summary = responses_reasoning_text(item)
                    .filter(|text| !text.trim().is_empty())
                    .context("本地 Chat 推理历史缺少可回放文本，无法转换为 Responses 消息")?;
                // Chat 合成项不是上游加密推理状态；以普通助手上下文保留文本，不伪造可复用状态。
                *item = json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": format!("Previous model reasoning summary:\n{summary}")
                    }]
                });
                let _ = crate::diagnostic_log::append_diagnostic_log(
                    "protocol_proxy.chat_reasoning_history_compat",
                    json!({ "reason": "synthetic_chat_reasoning_replayed_as_text" }),
                );
            }
            Some("custom_tool_call") => {
                let input = item
                    .get("input")
                    .and_then(Value::as_str)
                    .context("custom_tool_call 的 input 必须是字符串")?
                    .to_string();
                let object = item.as_object_mut().context("工具历史必须是对象")?;
                object.remove("input");
                object.insert("type".to_string(), json!("function_call"));
                object.insert(
                    "arguments".to_string(),
                    json!(json!({ "input": input }).to_string()),
                );
                normalize_responses_item_id(item);
            }
            Some("custom_tool_call_output") => {
                item["type"] = json!("function_call_output");
                normalize_responses_item_id(item);
            }
            _ => {}
        }
        Ok(())
    }

    fn is_custom_call(&self, item: &Value) -> bool {
        item.get("type").and_then(Value::as_str) == Some("function_call")
            && item
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| {
                    let namespace = item.get("namespace").and_then(Value::as_str).unwrap_or("");
                    self.custom_tools
                        .contains(&(namespace.to_string(), name.to_string()))
                })
    }

    fn restore_item(&self, item: &mut Value, incomplete: bool) -> anyhow::Result<bool> {
        if !self.is_custom_call(item) {
            return Ok(false);
        }
        let arguments = item.get("arguments").and_then(Value::as_str);
        let input = match arguments {
            Some(arguments) => decode_custom_input(arguments, incomplete)?,
            None if incomplete => String::new(),
            None => anyhow::bail!("custom 工具的 function_call 缺少 arguments"),
        };
        let object = item.as_object_mut().context("工具响应必须是对象")?;
        object.remove("arguments");
        object.insert("type".to_string(), json!("custom_tool_call"));
        object.insert("input".to_string(), json!(input));
        normalize_responses_item_id(item);
        Ok(true)
    }

    fn restore_response(&self, response: &mut Value) -> anyhow::Result<bool> {
        let incomplete = matches!(
            response.get("status").and_then(Value::as_str),
            Some("in_progress" | "failed" | "incomplete" | "cancelled")
        );
        let mut changed = false;
        if let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) {
            for item in output {
                changed |= self.restore_item(item, incomplete)?;
            }
        }
        Ok(changed)
    }

    pub fn restore_response_bytes(&self, bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut response: Value =
            serde_json::from_slice(bytes).context("Responses 工具兼容无法解析上游 JSON 响应")?;
        if self.restore_response(&mut response)? {
            Ok(serde_json::to_vec(&response)?)
        } else {
            Ok(bytes.to_vec())
        }
    }
}

fn decode_custom_input(arguments: &str, incomplete: bool) -> anyhow::Result<String> {
    if incomplete && arguments.is_empty() {
        return Ok(String::new());
    }
    let value = match serde_json::from_str::<Value>(arguments) {
        Ok(value) => value,
        Err(error) if incomplete && error.is_eof() => return Ok(String::new()),
        Err(_) => anyhow::bail!("custom 工具的 function arguments 不是有效 JSON"),
    };
    value
        .get("input")
        .and_then(Value::as_str)
        .map(str::to_string)
        .context("custom 工具的 function arguments 缺少字符串 input")
}

#[derive(Default)]
struct CustomCallState {
    client_item_id: String,
    arguments: String,
    emitted: String,
}

pub struct ResponsesToolSseConverter {
    context: ResponsesToolCompat,
    buffer: Vec<u8>,
    calls: BTreeMap<String, CustomCallState>,
}

impl ResponsesToolSseConverter {
    pub fn new(context: ResponsesToolCompat) -> Self {
        Self {
            context,
            buffer: Vec::new(),
            calls: BTreeMap::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut output = Vec::new();
        for byte in bytes {
            self.buffer.push(*byte);
            if self.buffer.ends_with(b"\n\n") || self.buffer.ends_with(b"\r\n\r\n") {
                output.extend(self.restore_block()?);
                self.buffer.clear();
            }
        }
        Ok(output)
    }

    pub fn finish(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.buffer.iter().all(u8::is_ascii_whitespace),
            "Responses 工具兼容收到未完成的 SSE 事件"
        );
        Ok(())
    }

    fn restore_block(&mut self) -> anyhow::Result<Vec<u8>> {
        let block = std::str::from_utf8(&self.buffer).context("Responses SSE 不是有效 UTF-8")?;
        let data = block
            .lines()
            .filter_map(|line| strip_sse_field(line, "data"))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() || data.trim() == "[DONE]" {
            return Ok(self.buffer.clone());
        }
        let mut event: Value =
            serde_json::from_str(&data).context("Responses 工具兼容无法解析 SSE JSON")?;
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| {
                block
                    .lines()
                    .find_map(|line| strip_sse_field(line, "event"))
            })
            .unwrap_or("")
            .to_string();
        let changed = match event_type.as_str() {
            "response.output_item.added" | "response.output_item.done" => {
                let mut changed = false;
                if let Some(item) = event.get_mut("item") {
                    if self.context.is_custom_call(item) {
                        let upstream_id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .context("custom 工具输出缺少 item id")?
                            .to_string();
                        let arguments = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let added = event_type == "response.output_item.added";
                        self.context.restore_item(item, added)?;
                        if added {
                            self.calls.insert(
                                upstream_id,
                                CustomCallState {
                                    client_item_id: item["id"]
                                        .as_str()
                                        .context("custom 工具缺少 item id")?
                                        .to_string(),
                                    emitted: item["input"]
                                        .as_str()
                                        .context("custom 工具缺少 input")?
                                        .to_string(),
                                    arguments,
                                },
                            );
                        }
                        changed = true;
                    }
                }

                changed
            }
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                let item_id = event.get("item_id").and_then(Value::as_str).unwrap_or("");
                if let Some(state) = self.calls.get_mut(item_id) {
                    event["item_id"] = json!(state.client_item_id);
                    if event_type.ends_with(".delta") {
                        let delta = event
                            .get("delta")
                            .and_then(Value::as_str)
                            .context("function_call_arguments.delta 缺少字符串 delta")?;
                        state.arguments.push_str(delta);
                        // JSON 包装完整前不把转义片段误当成原始工具输入发给 Codex。
                        let input = decode_custom_input(&state.arguments, true)?;
                        anyhow::ensure!(
                            input.starts_with(&state.emitted),
                            "custom 工具输入增量不一致"
                        );
                        event["type"] = json!("response.custom_tool_call_input.delta");
                        event["delta"] = json!(&input[state.emitted.len()..]);
                        state.emitted = input;
                    } else {
                        let arguments = event
                            .get("arguments")
                            .and_then(Value::as_str)
                            .context("function_call_arguments.done 缺少 arguments")?;
                        let input = decode_custom_input(arguments, false)?;
                        anyhow::ensure!(
                            input.starts_with(&state.emitted),
                            "custom 工具完成输入与增量不一致"
                        );
                        event
                            .as_object_mut()
                            .context("SSE 事件必须是对象")?
                            .remove("arguments");
                        event["type"] = json!("response.custom_tool_call_input.done");
                        event["input"] = json!(input);
                    }
                    true
                } else {
                    false
                }
            }
            "response.created"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete" => {
                if let Some(response) = event.get_mut("response") {
                    self.context.restore_response(response)?
                } else {
                    false
                }
            }
            _ => false,
        };
        if !changed {
            return Ok(self.buffer.clone());
        }
        let mut output = String::new();
        for line in block.lines() {
            if !line.is_empty()
                && strip_sse_field(line, "event").is_none()
                && strip_sse_field(line, "data").is_none()
            {
                output.push_str(line);
                output.push('\n');
            }
        }
        let event_name = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or(&event_type);
        output.push_str(&format!(
            "event: {event_name}\ndata: {}\n\n",
            serde_json::to_string(&event)?
        ));
        Ok(output.into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Value {
        json!({
            "model": "fixture-model",
            "tools": [
                {"type": "custom", "name": "apply_patch", "description": "Apply a patch",
                 "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"patch\""}},
                {"type": "namespace", "name": "functions", "tools": [
                    {"type": "custom", "name": "run_code", "format": {"type": "text"}},
                    {"type": "function", "name": "ordinary", "parameters": {"type": "object"}}
                ]}
            ],
            "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [
                {"type": "custom", "name": "apply_patch"},
                {"type": "function", "namespace": "functions", "name": "ordinary"}
            ]}
        })
    }

    fn context() -> ResponsesToolCompat {
        ResponsesToolCompat::prepare_request(&mut request())
            .unwrap()
            .unwrap()
    }

    fn custom_call(arguments: &str) -> Value {
        json!({
            "type": "function_call", "id": "fc_patch", "call_id": "call_patch",
            "name": "apply_patch", "arguments": arguments, "status": "completed"
        })
    }

    fn event(value: Value) -> String {
        format!(
            "event: {}\ndata: {value}\n\n",
            value["type"].as_str().unwrap()
        )
    }

    #[test]
    fn chat_reasoning_history_replays_after_client_serialization() {
        let response = crate::protocol_proxy::chat_completion_to_response(json!({
            "id": "chat_fixture",
            "choices": [{"message": {
                "role": "assistant",
                "reasoning_content": "Earlier model context.",
                "content": "Earlier answer."
            }}]
        })).unwrap();
        let mut reasoning = response["output"][0].clone();
        reasoning.as_object_mut().unwrap().remove("reasoning_content");
        for encrypted in [None, Some(Value::Null), Some(json!(""))] {
            let mut item = reasoning.clone();
            item["content"] = Value::Null;
            if let Some(encrypted) = encrypted {
                item["encrypted_content"] = encrypted;
            }
            let mut request = json!({
                "input": [item, response["output"][1]],
                "reasoning": {"effort": "high"}
            });
            assert!(ResponsesToolCompat::prepare_request(&mut request).unwrap().is_none());
            assert_eq!(request["input"][0], json!({
                "type": "message", "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Previous model reasoning summary:\nEarlier model context."
                }]
            }));
            assert_eq!(request["input"][1], response["output"][1]);
            assert_eq!(request["reasoning"]["effort"], "high");
            let once = request.clone();
            ResponsesToolCompat::prepare_request(&mut request).unwrap();
            assert_eq!(request, once);
        }
    }

    #[test]
    fn native_reasoning_and_encrypted_states_remain_unchanged() {
        let mut request = json!({
            "input": [
                {
                    "type": "reasoning", "id": "rs_native",
                    "summary": [{"type": "summary_text", "text": "Native summary"}],
                    "encrypted_content": null
                },
                {
                    "type": "reasoning", "id": "rs_resp_native",
                    "summary": [], "encrypted_content": "fixture-native-state"
                },
                {"type": "reasoning", "id": "rs_resp_", "summary": []},
                {"type": "compaction", "encrypted_content": "fixture-native-compaction"},
                {"type": "message", "role": "user", "content": "hello"}
            ]
        });
        let original = request.clone();
        assert!(ResponsesToolCompat::prepare_request(&mut request).unwrap().is_none());
        assert_eq!(request, original);
    }

    #[test]
    fn chat_reasoning_history_without_text_fails_explicitly() {
        let mut request = json!({
            "input": {
                "type": "reasoning", "id": "rs_resp_empty", "summary": [],
                "encrypted_content": null
            }
        });
        let error = ResponsesToolCompat::prepare_request(&mut request).unwrap_err();
        assert!(error.to_string().contains("本地 Chat 推理历史缺少可回放文本"));
    }

    #[test]
    fn custom_definitions_choices_and_history_use_responses_functions() {
        let mut request = request();
        let ordinary = request["tools"][1]["tools"][1].clone();
        let raw = "*** Begin Patch\n*** Add File: fixture.txt\n+hello \"世界\" 🦀\n*** End Patch";
        request["input"] = json!([
            {"type": "custom_tool_call", "id": "ctc_patch", "call_id": "call_patch",
             "name": "apply_patch", "input": raw},
            {"type": "custom_tool_call_output", "id": "ctco_patch", "call_id": "call_patch",
             "output": [{"type": "input_text", "text": "done"}]},
            {"type": "message", "role": "user", "content": "continue"}
        ]);
        let messages = request["input"][2].clone();
        ResponsesToolCompat::prepare_request(&mut request)
            .unwrap()
            .unwrap();
        assert_eq!(request["model"], "fixture-model");
        let tool = &request["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "apply_patch");
        assert!(
            tool.get("function").is_none(),
            "must stay in Responses format"
        );
        assert!(tool.get("format").is_none());
        assert_eq!(tool["parameters"]["required"], json!(["input"]));
        assert!(tool["description"].as_str().unwrap().contains("lark"));
        assert_eq!(request["tools"][1]["name"], "functions");
        assert_eq!(request["tools"][1]["tools"][0]["type"], "function");
        assert_eq!(request["tools"][1]["tools"][1], ordinary);
        assert_eq!(request["tool_choice"]["tools"][0]["type"], "function");
        assert_eq!(request["input"][0]["type"], "function_call");
        assert_eq!(request["input"][0]["id"], "fc_patch");
        assert_eq!(request["input"][0]["call_id"], "call_patch");
        let args: Value =
            serde_json::from_str(request["input"][0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["input"], raw);
        assert_eq!(request["input"][1]["type"], "function_call_output");
        assert_eq!(request["input"][1]["id"], "fco_patch");
        assert_eq!(
            request["input"][1]["output"],
            json!([{"type":"input_text","text":"done"}])
        );
        assert_eq!(request["input"][2], messages);
    }

    #[test]
    fn non_stream_tool_calls_round_trip_without_changing_native_fields() {
        let context = context();
        let raw = "print(\"hello\\n世界\")";
        let arguments = json!({"input": raw}).to_string();
        let original = json!({
            "id": "resp_test", "status": "completed", "usage": {"input_tokens": 23},
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
                custom_call(&arguments),
                {"type": "function_call", "id": "fc_normal", "name": "ordinary",
                 "namespace": "functions", "arguments": "{}"}
            ]
        });
        let bytes = context
            .restore_response_bytes(original.to_string().as_bytes())
            .unwrap();
        let response: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["id"], original["id"]);
        assert_eq!(response["usage"], original["usage"]);
        assert_eq!(response["output"][0], original["output"][0]);
        assert_eq!(response["output"][2], original["output"][2]);
        let restored = &response["output"][1];
        assert_eq!(restored["type"], "custom_tool_call");
        assert_eq!(restored["id"], "ctc_patch");
        assert_eq!(restored["input"], raw);
        assert!(restored.get("arguments").is_none());
        let mut replay = json!({"input": [restored.clone()]});
        ResponsesToolCompat::prepare_request(&mut replay).unwrap();
        assert_eq!(replay["input"][0], original["output"][1]);
        let unchanged = b"{ \"id\":\"resp_plain\", \"output\":[] }";
        assert_eq!(
            context.restore_response_bytes(unchanged).unwrap(),
            unchanged
        );
    }

    #[test]
    fn namespace_matching_does_not_rewrite_same_named_ordinary_functions() {
        let context = context();
        let mut response = json!({"output": [
            {"type":"function_call","id":"fc_code","call_id":"code",
             "namespace":"functions","name":"run_code","arguments":"{\"input\":\"1+1\"}"},
            {"type":"function_call","id":"fc_other","name":"run_code","arguments":"{}"}
        ]});
        let ordinary = response["output"][1].clone();
        context.restore_response(&mut response).unwrap();
        assert_eq!(response["output"][0]["type"], "custom_tool_call");
        assert_eq!(response["output"][0]["namespace"], "functions");
        assert_eq!(response["output"][0]["input"], "1+1");
        assert_eq!(response["output"][1], ordinary);
    }

    #[test]
    fn streaming_custom_calls_survive_fragmented_json_utf8_and_preserve_native_events() {
        let input = "*** Begin Patch\n+\"世界\" 🦀\\n\n*** End Patch";
        let arguments = json!({"input": input}).to_string();
        let mut added = custom_call("");
        added["status"] = json!("in_progress");
        let native = "id: keep\r\nevent: response.reasoning_summary_text.delta\r\ndata: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"native\"}\r\n\r\n";
        let mut sse =
            event(json!({"type":"response.output_item.added","output_index":0,"item":added}));
        sse.push_str(native);
        for character in arguments.chars() {
            sse.push_str(&event(json!({
                "type":"response.function_call_arguments.delta","item_id":"fc_patch",
                "output_index":0,"delta":character.to_string()
            })));
        }
        sse.push_str(&event(json!({
            "type":"response.function_call_arguments.done","item_id":"fc_patch",
            "output_index":0,"arguments":arguments
        })));
        sse.push_str(&event(json!({
            "type":"response.output_item.done","output_index":0,"item":custom_call(&arguments)
        })));
        sse.push_str(&event(json!({
            "type":"response.completed","response":{
                "id":"resp_test","status":"completed","usage":{"output_tokens":12},
                "output":[custom_call(&arguments)]
            }
        })));
        sse.push_str(
            "event: error\ndata: {\"type\":\"error\",\"message\":\"fixture upstream error\"}\n\n",
        );
        for chunk_size in [1, 2, 7, sse.len()] {
            let mut converter = ResponsesToolSseConverter::new(context());
            let mut bytes = Vec::new();
            for chunk in sse.as_bytes().chunks(chunk_size) {
                bytes.extend(converter.push_bytes(chunk).unwrap());
            }
            converter.finish().unwrap();
            let output = String::from_utf8(bytes).unwrap();
            assert!(
                output.contains(native),
                "native CRLF event must remain byte-identical"
            );
            assert!(output.ends_with("event: error\ndata: {\"type\":\"error\",\"message\":\"fixture upstream error\"}\n\n"));
            let events: Vec<Value> = output
                .lines()
                .filter_map(|line| strip_sse_field(line, "data"))
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            let deltas: String = events
                .iter()
                .filter(|event| event["type"] == "response.custom_tool_call_input.delta")
                .map(|event| event["delta"].as_str().unwrap())
                .collect();
            assert_eq!(deltas, input);
            assert_eq!(events[0]["item"]["type"], "custom_tool_call");
            assert_eq!(events[0]["item"]["id"], "ctc_patch");
            let done = events
                .iter()
                .find(|event| event["type"] == "response.custom_tool_call_input.done")
                .unwrap();
            assert_eq!(done["item_id"], "ctc_patch");
            assert_eq!(done["input"], input);
            let completed = events
                .iter()
                .find(|event| event["type"] == "response.completed")
                .unwrap();
            assert_eq!(completed["response"]["output"][0]["input"], input);
            assert_eq!(completed["response"]["usage"]["output_tokens"], 12);
        }
    }

    #[test]
    fn malformed_arguments_and_truncated_sse_fail_explicitly() {
        let context = context();
        for arguments in ["not json", "{}", "{\"input\":42}"] {
            let response = json!({"output":[custom_call(arguments)]});
            assert!(
                context
                    .restore_response_bytes(response.to_string().as_bytes())
                    .is_err()
            );
        }
        let mut converter = ResponsesToolSseConverter::new(context);
        converter
            .push_bytes(b"event: response.completed\ndata: {")
            .unwrap();
        assert!(converter.finish().is_err());
    }
}
