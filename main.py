# main.py
import argparse
from providers.provider_gateway import ProviderGateway
from langchain_integration.langgraph.routing_agent import run_routing_agent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["direct","graph"], default="graph")
    p.add_argument("--model", default="mistral7b")
    p.add_argument("--strategy", default="optimized")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("prompt")
    args = p.parse_args()

    gw = ProviderGateway()
    if args.mode == "direct":
        res = gw.generate({
            "prompt": args.prompt,
            "model_key": args.model,
            "strategy": args.strategy,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        })
        print(res.get("output",""))
        print("\n--- metrics:", res.get("metrics", {}))
        print("--- timings_ms:", res.get("timings_ms", {}))
    else:
        out = run_routing_agent(args.prompt, gateway=gw, verbose=True)
        print("\n=== OUTPUT ===\n")
        print(out.get("output",""))

if __name__ == "__main__":
    main()
