from typing import Tuple
import sys
import time
from nltk import WordNetLemmatizer


class TrieNode(object):
    """
    Our trie node implementation. Very basic. but does the job
    """

    def __init__(self, char: str):
        self.char = char
        self.children = []
        # Is it the last character of the word.`
        self.word_finished = False
        # How many times this character appeared in the addition process
        self.counter = 1
        # added parent node
        self.parent = None

    def pprint(self, indent="", last=True, stack=""):
        if indent != "":
            stack = stack + self.char

        sys.stdout.write(indent)
        if last:
            sys.stdout.write("┗╾")
            indent += "  "
        else:
            sys.stdout.write("┣╾")
            indent += "┃ "

        sys.stdout.write("{} ({})".format(self.char, self.counter))
        if self.word_finished:
            print(" - {}".format(stack))
        else:
            print()

        for i, c in enumerate(self.children):
            c.pprint(indent, i == len(self.children) - 1, stack)


def add(root, word: str):
    """
    Adding a word in the trie structure
    """
    node = root
    for char in word:
        found_in_child = False
        # Search for the character in the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found it, increase the counter by 1 to keep track that another
                # word has it as well
                child.counter += 1
                # And point the node to the child that contains this char
                node = child
                found_in_child = True
                break
        # We did not find it so add a new child
        if not found_in_child:
            new_node = TrieNode(char)
            node.children.append(new_node)
            new_node.parent = node
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a word.
    node.word_finished = True


def delete(root, prefix: str):
    """
        Delete Node at the last letter of prefix (if exists)
        To do this, we need to check first if prefix exists,
        then we need to determine the path to the end of the prefix
        and then pop the last char node
    """

    if find_prefix(root, prefix)[0]:
        node = root
        for char in prefix[:-1]:
            # Search through all the children of the present `node`
            for child in node.children:
                if child.char == char:
                    # We found the char existing in the child.
                    # Assign node as the child containing the char and break
                    node = child
                    break

        # Node at this point is the second to last character in trie, prune
        # via list revision
        node.children = [child for child in node.children if child.char != prefix[-1]]
        # Need to update the node.counter
        node.counter = len(node.children)


def find_prefix(root, prefix: str) -> Tuple[bool, int, str]:
    """
    Check and return
      1. If the prefix exists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    matched = ''
    possibilities = []
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, 0, matched
    for char in prefix:
        char_not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found the char existing in the child.
                char_not_found = False
                # Assign node as the child containing the char and break
                node = child
                matched += char
                if node.word_finished:
                    possibilities.append(matched)
                break
        # Return False anyway when we did not find a char.
        if char_not_found:
            if not possibilities:
                print(scoring_function(prefix, possibilities, back_track(node, level=2)))
            return False, 0, matched
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    print(node.char, node.word_finished, possibilities)
    return True, node.counter, matched


def get_node(node, level):
    while level:
        node = node.parent
        level -= 1
    return node


def back_track(node, level=1):
    node = get_node(node, level=level)
    return build_all_children(node)


def build_all_children(node):
    possibilities = []
    if not node.children:
        possibilities.append('')
        return possibilities

    for node in node.children:
        for s in build_all_children(node):
            possibilities.append(str(node.char) + s)
    return possibilities


def scoring_function(string, matched, possibilities):
    higest_matched = matched[-1]
    return [higest_matched[:-1] + str(x) for x in possibilities if len(x) + len(higest_matched[:-1]) <= len(string)] + matched


def read_file(filename):
    with open(filename, "r") as infile:
        data = infile.readlines()
    return data


if __name__ == "__main__":
    start = time.time()
    root = TrieNode('*')
    data = read_file("/Users/deepakpaudel/mycodes/stemmer/files/root")
    print(len(data))
    for d in data:
        d = d.strip().split("|")[0]
        add(root, d)
    print(time.time()-start)
    print(find_prefix(root, 'कविर'))

# खसम
# खात
# खाते
# खाक
# खाका
# खाकी
# खाकर
# खाक्सी
