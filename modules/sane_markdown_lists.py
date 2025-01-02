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
from typing import TYPE_CHECKING
import xml.etree.ElementTree as etree

from markdown import Extension
from markdown.blockprocessors import OListProcessor, ListIndentProcessor

if TYPE_CHECKING:  # pragma: no cover
    from .. import blockparser


class SaneListIndentProcessor(ListIndentProcessor):
    """ Process children of list items.

    Example

        * a list item
            process this part

            or this part

    """

    def __init__(self, *args):
        super().__init__(*args)
        self.INDENT_RE = re.compile(r'^(([ ]{%s})+)' % self.tab_length)

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' '*self.tab_length) and \
            not self.parser.state.isstate('detabbed') and \
            (parent.tag in self.ITEM_TYPES or
                (len(parent) and parent[-1] is not None and
                    (parent[-1].tag in self.LIST_TYPES)))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        level, sibling = self.get_level(parent, block)
        block = self.looseDetab(block, level)

        self.parser.state.set('detabbed')
        if parent.tag in self.ITEM_TYPES:
            # It's possible that this parent has a `ul` or `ol` child list
            # with a member.  If that is the case, then that should be the
            # parent.  This is intended to catch the edge case of an indented
            # list whose first member was parsed previous to this point
            # see `OListProcessor`
            if len(parent) and parent[-1].tag in self.LIST_TYPES:
                self.parser.parseBlocks(parent[-1], [block])
            else:
                # The parent is already a `li`. Just parse the child block.
                self.parser.parseBlocks(parent, [block])
        elif sibling.tag in self.ITEM_TYPES:
            # The sibling is a `li`. Use it as parent.
            self.parser.parseBlocks(sibling, [block])
        elif len(sibling) and sibling[-1].tag in self.ITEM_TYPES:
            # The parent is a list (`ol` or `ul`) which has children.
            # Assume the last child `li` is the parent of this block.
            if sibling[-1].text:
                # If the parent `li` has text, that text needs to be moved to a `p`
                # The `p` must be 'inserted' at beginning of list in the event
                # that other children already exist i.e.; a nested sub-list.
                p = etree.Element('p')
                p.text = sibling[-1].text
                sibling[-1].text = ''
                sibling[-1].insert(0, p)
            self.parser.parseChunk(sibling[-1], block)
        else:
            self.create_item(sibling, block)
        self.parser.state.reset()

    def get_level(self, parent: etree.Element, block: str) -> tuple[int, etree.Element]:
        """ Get level of indentation based on list level. """
        # Get indent level
        m = self.INDENT_RE.match(block)
        if m:
            indent_level = len(m.group(1))/self.tab_length
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
            length = self.tab_length
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
        """ Remove a tab from front of lines but allowing dedented lines. """
        lines = text.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith(' ' * self.tab_length * level):
                lines[i] = lines[i][self.tab_length * level:]
        return '\n'.join(lines)


class SaneOListProcessor(OListProcessor):
    """ Override `SIBLING_TAGS` to not include `ul` and set `LAZY_OL` to `False`. """

    SIBLING_TAGS = ['ol']
    """ Exclude `ul` from list of siblings. """
    LAZY_OL = False
    """ Disable lazy list behavior. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        self.RE = re.compile(r'^[ ]{0,%d}\d+\.[ ]+(.*)' % 1)
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}((\d+\.))[ ]+(.*)' % 1)
        # Detect indented (nested) items of either type
        self.INDENT_RE = re.compile(r'^[ ]{%d,%d}((\d+\.)|[*+-])[ ]+.*' %
                                    (2, self.tab_length * 2 - 1))

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
            if item.startswith(' '*self.tab_length):
                # Item is indented. Parse with last item as parent
                self.parser.parseBlocks(lst[-1], [item])
            else:
                # New item. Create `li` and parse with it as parent
                li = etree.SubElement(lst, 'li')
                self.parser.parseBlocks(li, [item])
        self.parser.state.reset()

    def get_items(self, block: str) -> list[str]:
        """ Break a block into list items. """
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                # This is a new list item
                # Check first item for the start index
                if not items and self.TAG == 'ol':
                    # Detect the integer value of first list item
                    INTEGER_RE = re.compile(r'(\d+)')
                    self.STARTSWITH = INTEGER_RE.match(m.group(1)).group()
                # Append to the list
                items.append(m.group(3))
            elif self.INDENT_RE.match(line):
                # This is an indented (possibly nested) item.
                if items[-1].startswith(' '*self.tab_length):
                    # Previous item was indented. Append to that item.
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                # This is another line of previous item. Append to that item.
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items


class SaneUListProcessor(SaneOListProcessor):
    """" Override `SIBLING_TAGS` to not include `ol`. """

    TAG: str = 'ul'
    SIBLING_TAGS = ['ul']
    """ Exclude `ol` from list of siblings. """

    def __init__(self, parser: blockparser.BlockParser):
        super().__init__(parser)
        # Detect an item (`1. item`). `group(1)` contains contents of item.
        self.RE = re.compile(r'^[ ]{0,%d}[*+-][ ]+(.*)' % 1)
        self.CHILD_RE = re.compile(r'^[ ]{0,%d}(([*+-]))[ ]+(.*)' % 1)


class SaneListExtension(Extension):
    """ Add sane lists to Markdown. """

    def extendMarkdown(self, md):
        """ Override existing Processors. """
        md.parser.blockprocessors.register(SaneListIndentProcessor(md.parser), 'indent', 90)
        md.parser.blockprocessors.register(SaneOListProcessor(md.parser), 'olist', 40)
        md.parser.blockprocessors.register(SaneUListProcessor(md.parser), 'ulist', 30)


def makeExtension(**kwargs):  # pragma: no cover
    return SaneListExtension(**kwargs)