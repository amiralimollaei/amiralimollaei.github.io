---
title: The journey of creating a chatbot that loves pancakes
description: >-
  How I reinvented "tool calling" to make a new kind of chatbot that can do virtually anything, multiple times.
author: amiralimollaei
date: 2025-07-18 20:03:00 +0330
categories: ["ai"]
tags: ["ai", "python", "LLM"]
pin: false
media_subpath: /assets/img/2025-07-18-writing-an-ai-chatbot-in-python/
---
> All the code in this article is published on my GitHub account at [amiralimollaei/pixi-bot](https://github.com/amiralimollaei/pixi-bot).

Sample interactions with the chatbot in Discord (for the rest of the article, I will refer to it as Pixi):

| Pancakes | World Domination | Cat Lover |
| :----------- |:-----------| :-----------|
| ![Content Cell](/Screenshot_20250703-004518_Discord.jpg)  | ![Content Cell](/Screenshot_20250626-010527_Discord.jpg) | ![Content Cell](/Screenshot_Discord.png) |

As you can see, this isn't exactly how ChatGPT answers your questions. Furthermore, you can see that the bot can do quite a lot of stuff that ChatGPT cannot, like sending GIFs, adding reactions, and sending multiple messages at once. To understand how this works, we first need to understand how *tool calling* can be used to integrate chatbots into almost any application.

## Tool Calling

You might be familiar with the concept of *tool calling* or have at least heard it a couple of times. OpenAI, the company that first introduced this concept, puts it this way:

> Function calling provides a powerful and flexible way for OpenAI models to interface with your code or external services and to connect the models to your own custom code to fetch data or take action.
>
> You can give the model access to your own custom code through function calling. Based on the system prompt and messages, the model may decide to call these functions — instead of (or in addition to) generating text or audio.
>
> – [OpenAI](https://platform.openai.com/docs/guides/function-calling)

Basically, it allows the model to call some pre-defined functions. This is exactly how many of ChatGPT's features work, like memory, image generation, web search, and its deep research mode.

### How does Tool Calling actually work?

I don't think there's any better explanation of how tool calling works than the following diagram from... you guessed it, OpenAI!

The following diagram shows an example of a function that fetches the real-time temperature of a location for the model:

![function-calling-diagram-steps.png](function-calling-diagram-steps.png)

Whenever you request something from the AI model, it checks to see if there are any tools that can help in resolving that request. If it finds any, it asks the client to call the appropriate tools. Then, our client executes the tool calls and sends their results back to the AI model. From there, the AI model can continue resolving your request based on the results provided.

However, this is not exactly how Pixi works. I spent most of my time coding Pixi when tool calling wasn't even a thing. Although tool calling is used for many of Pixi's new features, it is not used for adding reactions or sending multiple messages at a time.

## Inline Commands: Taking Actions Instantly

AI language models support two roles: the **Assistant** and the **User**. When you ask an AI model for the best cookie recipe ever, it basically sees your message as the **User** and starts writing a message as the **Assistant**. However, it must always go back and forth between these roles, meaning you can't have it generate more than one assistant response in a single turn. We can fix this using **inline commands**.

Inline commands are basically tools, with a few key differences: **they have at most one argument**, **they take immediate action**, and **they do not produce any output**. Since we don't need to send their results back to the AI model, **they are also instantaneous** compared to standard tools.

Below is a more complete comparison of tool calls and commands:

| | Tool Calling | Command Calling |
| ------------- | ------------- | ------------- |
| Speed | Seconds | Instant |
| Number of Arguments | Multiple* | Only one (Optional) |
| Use Case | Performing tasks that can sometimes fail | Taking actions that are not expected to fail |
| Output | Always produces an output | Never produces an output |
| Runtime Cost | More | Less |
| API Support | Sometimes | Always |

> *It is up to the API provider to limit the number of arguments a tool can have.

The implementation is quite simple. The following code processes the model's output stream for inline commands:

```python
# consumes commands and runs them automatically
def stream_commands(self, stream: Iterator):
    inside_command = 0
    command_str = ""
    for char in stream:
        result = None
        
        if char == "[":
            inside_command += 1
        if inside_command != 0:
            command_str += char
        else:
            result = char
        if char == "]":
            inside_command -= 1
            if inside_command == 0:
                _command_str = command_str[1:-1]
                seperator_idx = None
                if ":" in command_str:
                    seperator_idx = _command_str.index(":")
                command_data = None
                if seperator_idx:
                    command_name = _command_str[:seperator_idx].strip()
                    command_data = _command_str[seperator_idx+1:].strip()
                else:
                    command_name = _command_str
                command = self.commands.get(command_name.lower())
                if command is not None:
                    command(command_data)
                else:
                    raise NotImplementedError(f"The command `{command_name}` is not implemented.")
                command_str = ""
        if result:
            yield result
```

It works by finding parts of the model's output that follow the format of `[<name>]` or `[<name>: <value>]` and running the command that matches the `<name>`, passing it the `<value>` if present. Commands can be used multiple times, and since they have no output, they can run without additional API requests and be executed simultaneously while the model is generating more text and commands.

Some of the earliest commands in Pixi were `[SEND: <message>]` and `[NOTE: <notes>]`. Let's go over how they work:

- The `SEND` command is used to send a message in the chat. It is also used to *not* send anything. Pixi can sometimes choose to remain silent by simply not using the `SEND` command.
- The `NOTE` command is used for the model to write down its feelings, what it is going to do, how it will do it, and the reasoning behind its actions, **before doing anything**. This is important because the model needs to **plan** before it **takes action**. In my testing, this **significantly improves Pixi's overall performance**, and as a bonus, we can see the reasoning behind the scenes.

The bot uses these commands alongside others, like `[REACT: <emoji>]` to react to the user's message with an emoji. In Pixi, the `AsyncCommandManager` is responsible for handling commands and adding their descriptions to the system prompt so that the AI knows how to use them.

## Why Tools Are Still Very Useful

Even though commands can do a lot, there are many things they can't do because they don't produce output. For example, searching the web or querying data from a local database are tasks that require a result. Because of this, Pixi also implements standard tools. These are used to enable the bot to search Wikipedia, the Minecraft wiki, and local document databases. Tools are also used to find GIFs. Without them, none of these features would be possible without a significant performance degradation.

This is how I've implemented these features in Pixi:

```python
async def init_database_tool(self, database_name: str):
    if not self.enable_tool_calls:
        logging.warning("tried to initialize a database tool, but tool calls are disabled")
        return
    database_api = await DirectoryDatabase.from_directory(database_name)
    async def get_entry_as_str(entry_id: int):
        return json.dumps(asdict(await database_api.get_entry(entry_id)), ensure_ascii=False)
    async def search_database(instance: AsyncChatbotInstance, keyword: str):
        return [asdict(match) for match in await database_api.search(keyword)]
    self.register_tool(
        name=f"search_{database_name}_database",
        func=search_database,
        parameters=dict(
            type="object",
            properties=dict(
                keyword=dict(
                    type="string",
                    description=f"The search keyword to find matches in the text from the {database_name} database.",
                ),
            ),
            required=["keyword"],
            additionalProperties=False
        ),
        description=f"Searches the {database_name} database based on a keyword and returns entry metadata. You may use this function multiple times to find the specific information you're looking for."
    )
    async def query_database(instance: AsyncChatbotInstance, query: str, ids: str):
        if ids is None:
            return "no result, no ids specified"
        return await RetrievalAgent(
            model=self.helper_model,
            context=await asyncio.gather(*(get_entry_as_str(int(entry_id.strip())) for entry_id in ids.split(",")))
        ).retrieve(query)
    self.register_tool(
        name=f"query_{database_name}_database",
        func=query_database,
        parameters=dict(
            type="object",
            properties=dict(
                query=dict(
                    type="string",
                    description=f"A question or a statement that you want to find information about.",
                ),
                ids=dict(
                    type="string",
                    description=f"Comma-separated numerical entry IDs to fetch and query information from. Use `search_{database_name}_database` to obtain entry IDs based on a search term.",
                ),
            ),
            required=["query", "ids"],
            additionalProperties=False
        ),
        description=f"Runs an LLM agent to fetch and query the contents of the {database_name} database using entry IDs for finding relevant entries and a more detailed query for finding relevant information. Note that this will not return all the information that the page contains; you might need to use this command multiple times to get all the information out of a database entry."
    )
```

As you can see, we have two functions: `search_database` and `query_database`:

- `search_database` only returns snippets and titles of the entries that match the search term and sends them back to the AI model.
- `query_database` reads an entire database entry (which could be a long text like a full article or an entire book) and finds relevant information for a given query. It then sends only this relevant information back to the main AI model. This is accomplished using another, smaller LLM instance, which we call an **Agent**. Agents are useful for simplifying the work of the main AI model.

## Agents: What They Are and How They Are Used in Pixi

**Agents** are often smaller, simpler AI models that make the job of a larger AI model easier by performing small, focused tasks, often simultaneously. Think of agents as little robots in a factory controlled by a person; they accelerate the overall process by allowing the main AI to review their results rather than doing all the grunt work itself.

In our case, it also reduces API costs by a lot. In the `query_database` function, we pass a lot of text to a cheaper model (the `RetrievalAgent`). The `RetrievalAgent` finds and extracts only the relevant information, and we return only that concise result to the main AI model. This way, by reducing the data sent to the main model, we are **reducing API costs** and **improving its performance**. There is no longer a giant chunk of text appearing in the middle of our conversation every time the AI searches the database, so it can follow up with more coherent answers.

Below is the code for the `RetrievalAgent`:

```python
class RetrievalAgent:
    def __init__(self, model: Optional[str] = None, context: Optional[list[str]] = None):
        self.context = context or []
        self.model = model
        self.client = AsyncChatClient(model=model)
        self.system_prompt = "\n".join([
            "## You are a context retrieval agent",
            "",
            "Given a list of entries and a query, you must return any context that is relevant to the query.",
            "Write the response without losing any data, mention all the details; the less you summarize, the better.",
            "",
            "Output a JSON object with the following keys:",
            " - `relevant`: a list of all information that could possibly be used to answer the query in any way",
            " - `source`: a list of sources where the information was found, if applicable",
            " - `confidence`: a score value between 1 and 10 indicating how confident you are in the information provided",
            "",
            "Example output:",
            "```json",
            "{",
            "  \"relevant\": [\"Villagers can be cured from zombie villagers by using a splash potion of weakness and a golden apple.\"],",
            "  \"source\": [\"page_title:Villagers\"]",
            "  \"confidence\": 9",
            "}",
            "```",
        ])
        self.client.set_system(self.system_prompt)

    def to_dict(self) -> dict:
        return dict(context=self.context)

    @classmethod
    def from_dict(cls, data: dict) -> 'RetrievalAgent':
        context = data.get("context", [])
        return cls(context=context)

    def save_as(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def from_file(cls, file: str) -> 'RetrievalAgent':
        if os.path.isfile(file):
            with open(file, "rb") as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            inst = cls()
            inst.save_as(file)
            return inst

    def add_context(self, context: str):
        logging.debug(f"Adding context: {context}")
        self.context.append(context)

    async def retrieve(self, query: str) -> str:
        """
        Retrieves relevant information from the context based on a query.
        """
        logging.debug(f"Retrieving information for query: {query}")
        prompt = "\n".join([
            "Context:",
            "```json",
            json.dumps(self.context),
            "```",
            "",
            f"Query: \"{query}\"",
        ])
        response = ""
        async for char in self.client.stream_ask(prompt, temporal=True):
            response += char
        return response.strip()
```

## Beyond the Code: A Chatbot with Character

So, what does all this inline commands, agents, and tool calling have to do with a chatbot that loves pancakes? **Everything**. The goal was never just to build a functional bot, but to create one with **personality**.

The concepts I've explained here are the building blocks for that character. Inline commands give Pixi its quick, reactive nature, while tools and agents provide the opportunity to expand it's capabilities and help users in many meaningful ways.

This architecture is about more than just executing tasks; it’s about breathing life into lines of code, enabling a bot to express a thought, perform an action, and then share the result in a way that feels natural and dynamic. It turns a simple request-response model into a genuine conversation.

**All the code is available on my GitHub at [amiralimollaei/pixi-bot](https://github.com/amiralimollaei/pixi-bot).** Dive in, experiment, and let's push the boundaries of what conversational AI can be.
