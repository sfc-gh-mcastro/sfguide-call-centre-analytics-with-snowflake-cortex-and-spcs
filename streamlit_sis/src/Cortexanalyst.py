import streamlit as st
import json
import _snowflake
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from snowflake.snowpark.session import Session

# -- Snowflake session --
#session = get_active_session()

def cortexanlyst(session:Session):
    # -- Page Title --
    #st.title("Chat with your Structured Data :balloon:")
    st.subheader("Call Centre Insurance Assistant - Cortex Analyst")

    with st.expander("ℹ️ Instructions",expanded=True):
        st.markdown(
                """
                Enter a question and Snowflake Cortex Analyst will provide your an answer based on your strutured data.
                
                **Examples that you can use:**

                1. What are the distinct purposes of calls in 2023? Please include the count in descending order ?

                2. Can you derive the amount of calls that were resoled in the first call resolution aggregated by month ?

                3. Can you order the representatives per amount of positive and negative sentiment ?

                """
            )

    result = session.sql("select current_role();").collect()
    st.write(f"Current role: {result[0][0]}")

    API_ENDPOINT = "/api/v2/cortex/analyst/message"
    API_TIMEOUT = 50000  # in milliseconds

    # ------------------------------------------------------------------
    # Resetting the states
    #TODO: Keep different session states for cortex analyst and cortex search
    st.session_state.messages.clear()

    # Initialize st.session_state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []  # For our UI chat display
    if "analyst_conversation" not in st.session_state:
        st.session_state.analyst_conversation = []  # For sending to Cortex Analyst

    def get_sql_from_cortex_analyst():
        """
        Sends the entire analyst_conversation in the request to preserve context.
        TODO: get location of yaml file from a parameter
        """
        request_body = {
            "messages": st.session_state.analyst_conversation,  # full conversation so far
            "semantic_model_file": '@"LLMDEMO"."PUBLIC"."STREAMLIT_STAGE"/app/call_center_analytics.yaml',
        }
        
        resp = _snowflake.send_snow_api_request(
            "POST",
            API_ENDPOINT,
            {},
            {},
            request_body,
            None,
            API_TIMEOUT,
        )
        return json.loads(resp["content"])

    def parse_analyst_messages(analyst_output):
        """
        Extract text, sql, and suggestions items from a valid "analyst" output 
        or return an error dict if the shape is not as expected.
        """
        texts = []
        statements = []
        suggestions = []  # for suggestions
        
        # Case 1: analyst_output is a dict with "message" as dict
        if isinstance(analyst_output, dict) and "message" in analyst_output and isinstance(analyst_output["message"], dict):
            contents = analyst_output["message"].get("content", [])
            for item in contents:
                if item.get("type") == "text":
                    texts.append(item.get("text", "").strip())
                elif item.get("type") == "sql":
                    statements.append(item.get("statement", "").strip())
                elif item.get("type") == "suggestions":
                    suggestions.extend(item.get("suggestions", []))
            return {"status": "ok", "texts": texts, "statements": statements, "suggestions": suggestions}
        
        # Case 2: analyst_output is a dict with "message" as string
        elif isinstance(analyst_output, dict) and "message" in analyst_output and isinstance(analyst_output["message"], str):
            return {"status": "error", "error_message": analyst_output["message"]}

        # Case 3: analyst_output is a list
        elif isinstance(analyst_output, list) and analyst_output:
            raw_content = analyst_output[0].get("content", "").strip() if hasattr(analyst_output[0], "get") else str(analyst_output[0])
            return {"status": "error", "error_message": raw_content}

        # Case 4: fallback for any other case
        return {"status": "error", "error_message": f"Could not parse analyst output: {str(analyst_output)[:200]}"}

    def answer_question_using_analyst(query: str):
        """
        1) Append the user query to st.session_state.analyst_conversation.
        2) Call Cortex Analyst and parse the response.
        3) Append the analyst response (texts, SQL, and suggestions) to the conversation.
        4) If a SQL statement is returned, display the question interpretation (texts) and suggestions via st.info.
        (No st.info display occurs if no SQL is returned.)
        5) If no SQL is returned, yield the texts immediately.
        6) Otherwise, execute the SQL, display the SQL and results in expanders, and then generate the final answer from SQL results.
        7) Yield tokens from the streaming LLM response.
        
        This function always yields output (as a generator) so that the final answer can be streamed.
        """
        # STEP 1: Append user query
        st.session_state.analyst_conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": query}]
        })

        # STEP 2: Call Cortex Analyst
        with st.spinner("Calling Cortex Analyst to generate SQL..."):
            analyst_output = get_sql_from_cortex_analyst()
            parsed = parse_analyst_messages(analyst_output)

        if parsed["status"] == "error":
            yield f"**Error**: {parsed['error_message']}"
            return

        # Set flag indicating if SQL was returned
        has_sql = bool(parsed.get("statements"))
        st.session_state.has_sql = has_sql

        # STEP 3: Append analyst response to conversation
        analyst_message_content = []
        for txt in parsed.get("texts", []):
            analyst_message_content.append({"type": "text", "text": txt})
        for stmt in parsed.get("statements", []):
            analyst_message_content.append({"type": "sql", "statement": stmt})
        if parsed.get("suggestions"):
            analyst_message_content.append({"type": "suggestions", "suggestions": parsed["suggestions"]})
        st.session_state.analyst_conversation.append({
            "role": "analyst",
            "content": analyst_message_content
        })

        # STEP 4: Display interpretation and suggestions (only if SQL is present)
        if has_sql:
            for text in parsed.get("texts", []):
                st.info(text)
            if parsed.get("suggestions"):
                st.info("Suggestions:")
                for sug in parsed["suggestions"]:
                    st.info(f"- {sug}")

        # STEP 5: If no SQL, yield the analyst texts as final answer and finish
        if not has_sql:
            final_text = "\n".join(parsed.get("texts", []))
            yield final_text
            return

        # STEP 6: Process SQL - display SQL and execute it
        sql_statement = parsed["statements"][0]
        with st.expander("Generated SQL Statement"):
            st.code(sql_statement, language="sql")

        try:
            with st.spinner("Executing SQL..."):
                sql_output = session.sql(sql_statement.replace(";", ""))
        except Exception as e:
            yield f"**Error**: Failed to execute SQL. {e}. Please try again."
            return

        with st.expander("See SQL Results"):
            try:
                st.write(sql_output)
            except Exception as e:
                yield f"**Error**: Couldn't display SQL results. {e}"
                return

        # STEP 7: Generate final answer from SQL results using LLM streaming
        try:
            with st.spinner("Generating final answer from SQL results..."):
                df = sql_output.to_pandas()
                markdown_sql_output = df.to_markdown(index=False)
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that uses SQL output to answer questions."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The user asked: {query}\n\n"
                            f"The SQL results are:\n{markdown_sql_output}\n\n"
                            "Please answer the question concisely, without extra details."
                        )
                    }
                ]
                options = {"guardrails": True}
                response_generator = complete("claude-3-5-sonnet", messages, options=options, stream=True)
                
                # Yield tokens from the streaming LLM response as the final answer.
                for token in response_generator:
                    yield token
        except Exception as e:
            yield f"**Error**: Unable to generate final answer. {e}"
            return

    def display_messages():
        """
        Renders the entire UI chat from st.session_state.messages.
        """
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                st.chat_message("user").write(content)
            elif role == "assistant":
                st.chat_message("assistant", avatar="🤖").write(content)

    if st.button("Clear Conversation"):
        st.session_state.messages.clear()
        st.session_state.analyst_conversation.clear()
        if "has_sql" in st.session_state:
            del st.session_state["has_sql"]

    # Display previous conversation
    display_messages()

    # Prompt user for input
    user_input = st.chat_input("What are the distinct purposes of calls in 2023? Please include the count in descending order.")
    if user_input:
        # Display user's message
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate streaming tokens from the LLM answer generator.
        ai_answer_generator = answer_question_using_analyst(user_input)
        
        # Always display the final answer as a chat message (below st.info and expanders).
        final_text = st.chat_message("assistant", avatar="🤖").write_stream(ai_answer_generator)
        
        # Save the completed answer.
        st.session_state.messages.append({"role": "assistant", "content": final_text})
