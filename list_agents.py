"""
List all saved agents and their metadata.

Shows which agents exist, what environments they trained in,
and their learning history.
"""

import os
import torch
from pathlib import Path

def get_agent_info(model_path: str):
    """Extract metadata from a model file."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        return {
            'agent_name': checkpoint.get('agent_name', 'Unknown'),
            'environment': checkpoint.get('environment', 'Unknown'),
            'episode': checkpoint.get('episode', 'Unknown'),
            'file': os.path.basename(model_path),
            'size_kb': os.path.getsize(model_path) / 1024
        }
    except Exception as e:
        return {
            'agent_name': 'Error',
            'environment': 'Error',
            'episode': f'Error: {e}',
            'file': os.path.basename(model_path),
            'size_kb': 0
        }

def main():
    model_dirs = ['models', 'models_infinite']
    
    print("=" * 80)
    print("🤖 AGENT DIRECTORY")
    print("=" * 80)
    
    all_agents = {}
    
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        print(f"\n📁 {model_dir}/")
        print("-" * 80)
        
        # Find all .pth files
        model_files = list(Path(model_dir).glob("*.pth"))
        
        if not model_files:
            print("   (no agents found)")
            continue
        
        # Group by agent name
        agents_in_dir = {}
        for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
            info = get_agent_info(str(model_file))
            agent_name = info['agent_name']
            
            if agent_name not in agents_in_dir:
                agents_in_dir[agent_name] = []
            agents_in_dir[agent_name].append(info)
            
            # Also add to global list
            if agent_name not in all_agents:
                all_agents[agent_name] = []
            all_agents[agent_name].append(info)
        
        # Display agents
        for agent_name, checkpoints in sorted(agents_in_dir.items()):
            print(f"\n🤖 {agent_name}")
            for checkpoint in sorted(checkpoints, key=lambda x: x['episode'] if isinstance(x['episode'], int) else 0, reverse=True)[:5]:  # Show latest 5
                env = checkpoint['environment']
                ep = checkpoint['episode']
                file = checkpoint['file']
                size = checkpoint['size_kb']
                print(f"   📊 Env: {env:20} | Ep: {str(ep):6} | {file:40} ({size:6.1f} KB)")
            if len(checkpoints) > 5:
                print(f"   ... and {len(checkpoints) - 5} more checkpoints")
    
    print("\n" + "=" * 80)
    print(f"📈 Summary: {len(all_agents)} unique agents found")
    print("=" * 80)
    
    # Show agent summary
    if all_agents:
        print("\n🧬 Agents by name:")
        for agent_name in sorted(all_agents.keys()):
            checkpoints = all_agents[agent_name]
            envs = set(c['environment'] for c in checkpoints if c['environment'] != 'Unknown')
            latest = max(checkpoints, key=lambda x: x['episode'] if isinstance(x['episode'], int) else 0)
            print(f"   {agent_name:20} | {len(checkpoints):3} checkpoints | Latest env: {latest['environment']} | Latest ep: {latest['episode']}")

if __name__ == "__main__":
    main()

