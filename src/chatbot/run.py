"""
Interactive CLI runner for the Enkay Investment Advisory chatbot.

Usage:
    python -m src.chatbot.run

Requires OPENAI_API_KEY environment variable to be set.
"""

import asyncio
import sys

from agents import Runner, InputGuardrailTripwireTriggered


async def main():
    # Import here so module-level data loading happens after cwd is correct
    from src.chatbot.agent import investment_agent

    print("=" * 60)
    print("  Enkay Investment Advisory Chatbot")
    print("  Powered by OpenAI Agents SDK")
    print("=" * 60)
    print()
    print("Ask me anything about mutual funds, brokerage,")
    print("fund rankings, AUM analysis, or investment strategy.")
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            result = await Runner.run(
                investment_agent,
                conversation_history,
                max_turns=15,
            )

            response = result.final_output
            print(f"\nAdvisor: {response}")

            # Update conversation history for multi-turn
            conversation_history = result.to_input_list()

        except InputGuardrailTripwireTriggered:
            print(
                "\nAdvisor: I can only help with questions related to mutual fund "
                "investing, brokerage analysis, fund rankings, portfolio strategy, "
                "and investment advisory. Please ask something related to these topics."
            )
            # Remove the off-topic message from history
            conversation_history.pop()

        except Exception as e:
            print(f"\nError: {e}")
            # Remove failed message from history
            if conversation_history:
                conversation_history.pop()


if __name__ == "__main__":
    asyncio.run(main())
