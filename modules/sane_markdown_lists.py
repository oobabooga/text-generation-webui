# Code based on the Sane List Extension for Python-Markdown
# =======================================

# Modify the behavior of Lists in Python-Markdown to act in a sane manner.

# See https://Python-Markdown.github.io/extensions/sane_lists
# for documentation.

# Original code Copyright 2011 [Waylan Limberg](http://achinghead.com)

# All changes Copyright 2011-2014 The Python Markdown Project

# License: [BSD](https://opensource.org/licenses/bsd-license.php)

"""
Modify the behavior of Lists in Python-Markdown to act in a sane manner.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING

from markdown import Extension
from markdown.blockparser import BlockParser
from markdown.blockprocessors import (
    ListIndentProcessor,
    OListProcessor,
    ParagraphProcessor
)

if TYPE_CHECKING:  # pragma: no cover
    from markdown import blockparser


# The min. number of added leading spaces needed to start a nested list
MIN_NESTED_LIST_INDENT = 2
assert MIN_NESTED_LIST_INDENT > 1, "'MIN_NESTED_LIST_INDENT' must be > 1"


class SaneListIndentProcessor(ListIndentProcessor):
    """ Process children of list items.

    Example

        * a list item
            process this part

            or this part

    """

    def __init__(self, *args):
        super().__init__(*args)
        self.INDENT_RE = re.compile(r'^(([ ])+)')

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' ' * MIN_NESTED_LIST_INDENT) and \
            not self.parser.state.isstate('detabbed') and \
            (parent.tag in self.ITEM_TYPES or
                (len(parent) and parent[-1] is not None and
                    (parent[-1].tag in self.LIST_TYPES)))

    def get_level(self, parent: etree.Element, block: str) -> tuple[int, etree.Element]:
        """ Get level of indentation based on list level. """
        # Get indent level
        m = self.INDENT_RE.match(block)
        if m:
            indent_level = len(m.group(1)) / MIN_NESTED_LIST_INDENT
        else:
            indent_level = 0
        if self.parser.state.isstate('list'):
            # We're in a tight-list - so we already are at correct parent.
            level = 1
        else:
            # We're in a loose-list - so we need to find parent.
            level = 0
        # Step through children of tree to find matching indent level.
        while indent_level > level:
            child = self.lastChild(parent)
            if (child is not None and
                    (child.tag in self.LIST_TYPES or child.tag in self.ITEM_TYPES)):
                if child.tag in self.LIST_TYPES:
                    level += 1
                parent = child
            else:
                # No more child levels. If we're short of `indent_level`,
                # we have a code block. So we stop here.
                break
        return level, parent

    def detab(self, text: str, length: int | None = None) -> tuple[str, str]:
        """ Remove a tab from the front of each line of the given text. """
        if length is None:
            length = MIN_NESTED_LIST_INDENT
        newtext = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith(' ' * length):
                newtext.append(line[length:])
            elif not line.strip():
                newtext.append('')
            else:
                break
        return '\n'.join(newtext), '\n'.join(lines[len(newtext):])

    def looseDetab(self, text: str, level: int = 1) -> str:
        """ Remove indentation from front of lines but allowing dedented lines. """
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' ' * MIN_NESTED_LIST_INDENT * level):
                lines[i] = lines[i][MIN_NESTED_LIST_INDENT * level:]
        return '\n'.join(lines)


class SaneOListProcessor(OListProcessor):
    """ Override `SIBLING_TAGS` to not include `ul` and set `LAZY_OL` to `False`. """

    SIBLING_TAGS = ['ol']
    """ Exclude `ul` from list of siblings. """
    LAZY_OL = False
    """ Disable lazy list behavior. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        # This restriction stems from the 'CodeBlockProcessor' class,
        # which automatically matches blocks with an indent = self.tab_length
        max_list_start_indent = self.tab_length - 1
        # Detect an item (e.g., `1. item`)
        self.RE = re.compile(r'^[ ]{0,%d}[\*_]{0,2}\d+\.[ ]+(.*)' % max_list_start_indent)
        # Detect items on secondary lines. they can be of either list type.
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}([\*_]{0,2})((\d+\.))[ ]+(.*)' % (MIN_NESTED_LIST_INDENT - 1))
        # Detect indented (nested) items of either type
        self.INDENT_RE = re.compile(r'^[ ]{%d,%d}[\*_]{0,2}((\d+\.)|[*+-])[ ]+.*' %
                                    (MIN_NESTED_LIST_INDENT, self.tab_length * 2 - 1))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        # Check for multiple items in one block.
        items = self.get_items(blocks.pop(0))
        sibling = self.lastChild(parent)

        if sibling is not None and sibling.tag in self.SIBLING_TAGS:
            # Previous block was a list item, so set that as parent
            lst = sibling
            # make sure previous item is in a `p` - if the item has text,
            # then it isn't in a `p`
            if lst[-1].text:
                # since it's possible there are other children for this
                # sibling, we can't just `SubElement` the `p`, we need to
                # insert it as the first item.
                p = etree.Element('p')
                p.text = lst[-1].text
                lst[-1].text = ''
                lst[-1].insert(0, p)
            # if the last item has a tail, then the tail needs to be put in a `p`
            # likely only when a header is not followed by a blank line
            lch = self.lastChild(lst[-1])
            if lch is not None and lch.tail:
                p = etree.SubElement(lst[-1], 'p')
                p.text = lch.tail.lstrip()
                lch.tail = ''

            # parse first block differently as it gets wrapped in a `p`.
            li = etree.SubElement(lst, 'li')
            self.parser.state.set('looselist')
            firstitem = items.pop(0)
            self.parser.parseBlocks(li, [firstitem])
            self.parser.state.reset()
        elif parent.tag in ['ol', 'ul']:
            # this catches the edge case of a multi-item indented list whose
            # first item is in a blank parent-list item:
            #     * * subitem1
            #         * subitem2
            # see also `ListIndentProcessor`
            lst = parent
        else:
            # This is a new list so create parent with appropriate tag.
            lst = etree.SubElement(parent, self.TAG)
            # Check if a custom start integer is set
            if not self.LAZY_OL and self.STARTSWITH != '1':
                lst.attrib['start'] = self.STARTSWITH

        self.parser.state.set('list')
        # Loop through items in block, recursively parsing each with the
        # appropriate parent.
        for item in items:
            if item.startswith(" " * MIN_NESTED_LIST_INDENT):
                # Item is indented. Parse with last item as parent
                self.parser.parseBlocks(lst[-1], [item])
            else:
                # New item. Create `li` and parse with it as parent
                li = etree.SubElement(lst, 'li')
                self.parser.parseBlocks(li, [item])
        self.parser.state.reset()

    def looseDetab(self, text: str, indent_length: int, level: int = 1) -> str:
        """ Remove indentation from front of lines but allowing dedented lines. """
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' ' * indent_length * level):
                lines[i] = lines[i][indent_length * level:]
        return '\n'.join(lines)

    def get_items(self, block: str) -> list[str]:
        """ Break a block into list items. """
        # If first level of list is indented, remove that indentation
        if (indent_len := len(block) - len(block.lstrip())) > 0:
            block = self.looseDetab(block, indent_len)
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                # This is a new list item
                # Check first item for the start index
                if not items:
                    # Detect the integer value of first list item
                    INTEGER_RE = re.compile(r'(\d+)')
                    self.STARTSWITH = INTEGER_RE.match(m.group(2)).group()
                # Append to the list
                items.append(m.group(1) + m.group(4))
            elif self.INDENT_RE.match(line):
                # This is an indented (possibly nested) item.
                if items[-1].startswith(' ' * MIN_NESTED_LIST_INDENT):
                    # Previous item was indented. Append to that item.
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                # This is another line of previous item. Append to that item.
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items


class SaneUListProcessor(SaneOListProcessor):
    """ Override `SIBLING_TAGS` to not include `ol`. """

    TAG: str = 'ul'
    SIBLING_TAGS = ['ul']
    """ Exclude `ol` from list of siblings. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        # Detect an item (e.g., `- item` or `+ item` or `* item`).
        max_list_start_indent = self.tab_length - 1
        self.RE = re.compile(r'^[ ]{0,%d}[*+-][ ]+(.*)' % max_list_start_indent)
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}(([*+-]))[ ]+(.*)' % (MIN_NESTED_LIST_INDENT - 1))

    def get_items(self, block: str) -> list[str]:
        """ Break a block into list items. """
        # If first level of list is indented, remove that indentation
        if (indent_len := len(block) - len(block.lstrip())) > 0:
            block = self.looseDetab(block, indent_len)
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                # Append to the list
                items.append(m.group(3))
            elif self.INDENT_RE.match(line):
                # This is an indented (possibly nested) item.
                if items[-1].startswith(' ' * MIN_NESTED_LIST_INDENT):
                    # Previous item was indented. Append to that item.
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                # This is another line of previous item. Append to that item.
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items


class SaneParagraphProcessor(ParagraphProcessor):
    """ Process Paragraph blocks. """

    def __init__(self, parser: BlockParser):
        super().__init__(parser)
        max_list_start_indent = self.tab_length - 1
        self.LIST_RE = re.compile(r"\s{2}\n(\s{0,%d}[\d+*-])" % max_list_start_indent)

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        if block.strip():
            # Not a blank block. Add to parent, otherwise throw it away.
            if self.parser.state.isstate('list'):
                # The parent is a tight-list.
                #
                # Check for any children. This will likely only happen in a
                # tight-list when a header isn't followed by a blank line.
                # For example:
                #
                #     * # Header
                #     Line 2 of list item - not part of header.
                sibling = self.lastChild(parent)
                if sibling is not None:
                    # Insert after sibling.
                    if sibling.tail:
                        sibling.tail = '{}\n{}'.format(sibling.tail, block)
                    else:
                        sibling.tail = '\n%s' % block
                else:
                    # Append to parent.text
                    if parent.text:
                        parent.text = '{}\n{}'.format(parent.text, block)
                    else:
                        parent.text = block.lstrip()
            else:
                # Check if paragraph contains a list
                next_list_block = None
                if list_match := self.LIST_RE.search(block):
                    list_start = list_match.end() - len(list_match.group(1))
                    next_list_block = block[list_start:]
                    block = block[:list_start]

                # Create a regular paragraph
                p = etree.SubElement(parent, 'p')
                p.text = block.lstrip()

                # If a list was found, parse its block separately with the paragraph as the parent
                if next_list_block:
                    self.parser.parseBlocks(p, [next_list_block])


class SaneListExtension(Extension):
    """ Add sane lists to Markdown. """

    def extendMarkdown(self, md):
        """ Override existing Processors. """
        md.parser.blockprocessors.register(SaneListIndentProcessor(md.parser), 'indent', 90)
        md.parser.blockprocessors.register(SaneOListProcessor(md.parser), 'olist', 40)
        md.parser.blockprocessors.register(SaneUListProcessor(md.parser), 'ulist', 30)
        md.parser.blockprocessors.register(SaneParagraphProcessor(md.parser), 'paragraph', 10)


def makeExtension(**kwargs):  # pragma: no cover
    return SaneListExtension(**kwargs)
