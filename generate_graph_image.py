#!/usr/bin/env python3
"""
Script to generate and save the LangGraph visualization as an image
"""

import os
import sys

# Add project root for imports
sys.path.append(os.path.dirname(__file__))

from Agent_AI.agent import app

def main():
    print("Generating graph visualization...")
    
    try:
        # Get the graph
        graph = app.get_graph()
        
        # Generate the image
        print("Creating Mermaid diagram...")
        image_data = graph.draw_mermaid_png()
        
        # Save the image
        output_path = "agent_graph.png"
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        print(f"✅ Graph image saved as: {output_path}")
        print(f"📁 File location: {os.path.abspath(output_path)}")
        
        # Display the graph structure
        print("\n📊 Graph Structure:")
        print(graph)
        
    except Exception as e:
        print(f"❌ Error generating graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
