from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import re

@dataclass
class Node:
    backend_node_id: str
    tag: str
    text_content: str = ""
    aria_label: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    is_interactive: bool = False
    attributes: Dict[str, str] = field(default_factory=dict)

class BackendNodeParser:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.interactive_tags = {'button', 'a', 'input', 'select', 'textarea'}

    def is_interactive(self, tag):
        if not hasattr(tag, 'attrs'):
            return False
        return (
            tag.name in self.interactive_tags or
            tag.get('role') in {'button', 'link'} or
            tag.get('onclick') is not None or
            any(attr for attr in tag.attrs if attr.startswith('on'))  # Check for any event handlers
        )

    def parse_html(self, html_content: str):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            self._process_node(soup)
            return self.nodes
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return {}

    def _process_node(self, element, parent_id: Optional[str] = None):
        if element is None:
            return None

        # Handle text nodes
        if isinstance(element, str):
            return None

        node_id = element.get('backend_node_id')
        if not node_id:
            # Process children even if current node doesn't have backend_node_id
            for child in element.children:
                if not isinstance(child, str):
                    self._process_node(child, parent_id)
            return None

        # Create node
        text_content = ""
        if hasattr(element, 'get_text'):
            text_content = ' '.join(element.get_text().split())  # Normalize whitespace

        node = Node(
            backend_node_id=node_id,
            tag=element.name if hasattr(element, 'name') else 'text',
            text_content=text_content,
            aria_label=element.get('aria-label', element.get('aria_label')),  # Check both formats
            parent_id=parent_id,
            is_interactive=self.is_interactive(element),
            attributes={k: v for k, v in element.attrs.items() if k != 'backend_node_id'}
        )

        # Process children
        for child in element.children:
            child_id = self._process_node(child, node_id)
            if child_id and child_id not in node.children_ids:
                node.children_ids.append(child_id)

        self.nodes[node_id] = node
        return node_id

    def get_node_context(self, node_id: str, max_depth: int = 3) -> str:
        if node_id not in self.nodes:
            return ""

        context = set()  # Using set to avoid duplicates
        node = self.nodes[node_id]

        # Get parent context
        current = node
        depth = 0
        while current.parent_id and depth < max_depth:
            parent = self.nodes.get(current.parent_id)
            if parent and parent.text_content.strip():
                context.add(parent.text_content.strip())
            if not parent:
                break
            current = parent
            depth += 1

        # Get children context
        def get_children_text(node_id: str, current_depth: int):
            if current_depth >= max_depth:
                return
            node = self.nodes.get(node_id)
            if not node:
                return
            for child_id in node.children_ids:
                child = self.nodes.get(child_id)
                if child and child.text_content.strip():
                    context.add(child.text_content.strip())
                get_children_text(child_id, current_depth + 1)

        get_children_text(node_id, 0)
        return " ".join(context)

    def create_rag_index(self) -> Dict[str, Dict]:
        index = {}
        for node_id, node in self.nodes.items():
            if node.is_interactive:
                context = self.get_node_context(node_id)
                index[node_id] = {
                    'tag': node.tag,
                    'text_content': node.text_content,
                    'context': context,
                    'aria_label': node.aria_label,
                    'attributes': node.attributes,
                    'parent_id': node.parent_id,
                    'children_ids': node.children_ids
                }
        return index


def test_parser():
    # Test cases
    test_cases = [
        # Test 1: Empty HTML
        "",
        
        # Test 2: Invalid HTML
        "<invalid>",
        
        # Test 3: No backend node IDs
        "<div><button>Click me</button></div>",
        
        # Test 4: Nested structure with some missing backend node IDs
        """
        <div backend_node_id="1">
            <div>
                <button backend_node_id="2">Click</button>
            </div>
            <a backend_node_id="3" aria-label="Home">Home</a>
        </div>
        """,
        
        # Test 5: Text nodes and whitespace handling
        """
        <div backend_node_id="1">
            Some text
            <span backend_node_id="2">
                More    text    with    spaces
            </span>
        </div>
        """
    ]
    
    parser = BackendNodeParser()
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        nodes = parser.parse_html(test_case)
        rag_index = parser.create_rag_index()
        print(f"Nodes found: {len(nodes)}")
        print(f"Interactive elements: {len(rag_index)}")
        for node_id, info in rag_index.items():
            print(f"  Node {node_id}: {info['tag']} - '{info['text_content']}'")

if __name__ == "__main__":
    test_parser()