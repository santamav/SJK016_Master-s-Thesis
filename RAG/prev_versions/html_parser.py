from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set


@dataclass
class Node:
    backend_node_id: str
    tag: str
    text_content: str = ""
    aria_label: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)

class NodeParser:
    def __init__(self):
        self.nodes = Dict[str, Node] = {}
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