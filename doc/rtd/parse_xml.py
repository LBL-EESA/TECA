#!/usr/bin/env python3

import os
import xml.etree.ElementTree as ET
from collections import defaultdict


class TECA_Tree(object):
    def __init__(
            self, xml_dir='_build/xml',
            index_file='index.xml', kinds=['class']):
        """Construct TECA ('class' for now) Tree

        :param index_file: Doxygen's index.xml path
        :type index_file: str
        """
        self.index_file = index_file
        self.xml_dir = xml_dir
        self.kinds = kinds

        xml_tree = ET.parse(os.path.join(xml_dir, index_file))
        self.xml_root = xml_tree.getroot()

        self.nodes = defaultdict(None)
        self.directory_structure = None

        self.meta_info = {
            'alg': {
                'full_name': 'Algorithms',
                'description': 'TECA\'s suite of algorithms that can \
                                be inserted in functional pipelines'
            },
            'core': {
                'full_name': 'Core',
                'description': 'TECA\'s core components'
            },
            'data': {
                'full_name': 'Data',
                'description': 'TECA\'s data structures'
            },
            'io': {
                'full_name': 'I/O',
                'description': 'TECA\'s I/O components to read datasets \
                                efficiently'
            }
        }

        self.get_components()

        self.generate_file_hierarchy()

        self.rescue_every_family()
        self.generate_class_hierarchy()

        self.generate_api_pages(output_dir='_build/rst')

    def get_first_dir(self, trunk_dict):
        for key, _ in trunk_dict.items():
            if key != 'files':
                return key

    def get_compound_refid(self, name, kind='file'):
        for teca_element in self.xml_root.findall('compound'):
            if teca_element.get('kind') == kind:
                if name == teca_element.find('name').text:
                    return teca_element.get('refid')

    def get_components(self):
        for teca_element in self.xml_root.findall('compound'):
            kind = teca_element.get('kind')
            if kind in self.kinds:
                name = teca_element.find('name').text
                if '::' not in name:
                    node = self.Node(
                        teca_element.get('refid'),
                        name,
                        self.xml_dir
                        )

                    if node.location:
                        self.nodes[teca_element.get('refid')] = node

    def rescue_every_family(self):
        for node_refid, node in self.nodes.items():
            node.find_family(self.nodes)

            self.nodes[node_refid] = node

    def generate_class_hierarchy(self, output_dir='_build/rst'):
        def structure_class_hierarchy(
                node, element_type='Class', lastChild=False):
            html = ''
            if lastChild:
                html += '<li class="lastChild">'
            else:
                html += '<li>'

            html += element_type + ' <a href="' + node.refid + '.html">'
            html += node.name + '</a>'

            if node.children:
                html += '<ul>'
                children_len_minus_one = len(node.children) - 1
                for i, child_node in enumerate(node.children):
                    if i == children_len_minus_one:
                        html += structure_class_hierarchy(
                                    child_node, lastChild=True)
                    else:
                        html += structure_class_hierarchy(child_node)
                html += '</ul>'
            html += '</li>'
            return html

        rst = '\nClass Hierarchy\n~~~~~~~~~~~~~~~\n\n.. raw:: html\n\n   '

        html = '<ul class="treeView" id="class-treeView">'
        html += '<li><ul class="collapsibleList">'

        root_nodes = []
        for _, node in self.nodes.items():
            if node.parent is None and ('::' not in node.name):
                root_nodes.append(node)

        for i, root_node in enumerate(root_nodes):
            if i == len(root_nodes) - 1:
                html += structure_class_hierarchy(root_node, lastChild=True)
            else:
                html += structure_class_hierarchy(root_node)

        html += '</ul></li></ul>'

        rst += html + '\n\n.. end raw html for treeView\n'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(
                    output_dir, 'generated_rtd_class_hierarchy.rst'),
                  'w') as f:
            f.write(rst)

    def generate_file_hierarchy(self, output_dir='_build/rst'):
        def structure_file_hierarchy(name, trunk, lastChild=False):
            html = ''
            if lastChild:
                html += '<li class="lastChild">'
            else:
                html += '<li>'

            html += name

            trunk_len_minus_one = len(trunk.items()) - 1
            for i, (key, value) in enumerate(trunk.items()):
                html += '<ul>'

                if key == 'files':
                    files_len_minus_one = len(value) - 1
                    for j, file in enumerate(value):
                        if j == files_len_minus_one:
                            html += '<li class="lastChild">'
                        else:
                            html += '<li>'
                        html += '<a href="' + file[2] + '.html">'
                        html += file[0] + '</a>'
                        html += '</li>'
                else:
                    if i == trunk_len_minus_one:
                        html += structure_file_hierarchy(
                                    key, trunk[key], lastChild=True)
                    else:
                        html += structure_file_hierarchy(key, trunk[key])
                html += '</ul>'
            html += '</li>'

            return html

        trunk = defaultdict(dict, (('files', []),))

        def attach(location, trunk, refid):
            parts = location.split('/', 1)
            if len(parts) == 1:  # branch is a file
                trunk['files'].append((
                        parts[0],
                        refid,
                        self.get_compound_refid(parts[0]) + '_source')
                    )
            else:
                directory, others = parts
                if directory not in trunk:
                    trunk[directory] = defaultdict(dict, (('files', []),))
                attach(others, trunk[directory], refid)

        rst = '\nFile Hierarchy\n~~~~~~~~~~~~~~~\n\n.. raw:: html\n\n   '

        html = '<ul class="treeView" id="file-treeView">'
        html += '<li><ul class="collapsibleList">'

        for _, node in self.nodes.items():
            attach(node.location, trunk, node.refid)

        trunk['TECA'] = trunk.pop(self.get_first_dir(trunk))
        self.directory_structure = trunk

        first_dir = 'TECA'
        html += structure_file_hierarchy(
                    first_dir, trunk[first_dir], lastChild=True)

        html += '</ul></li></ul>'

        rst += html + '\n\n.. end raw html for treeView\n'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(
                    output_dir, 'generated_rtd_file_hierarchy.rst'),
                  'w') as f:
            f.write(rst)

    def generate_api_pages(self, output_dir='_build/rst'):
        first_dir = 'TECA'
        trunk = self.directory_structure[first_dir]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generated_files = []

        for key, value in trunk.items():
            if key == 'files':
                continue

            page_name = self.meta_info[key]['full_name']

            rst = ''
            rst += '\n.. _' + page_name + ':\n'
            rst += '\n' + page_name + '\n'
            rst += '-' * len(page_name) + '\n\n'
            rst += self.meta_info[key]['description'] + '. '
            rst += '(For more details, click on the class name) \n\n'

            rst += '.. csv-table:: TECA Classes\n'
            rst += '   :header: "Class", "Description"\n'
            rst += '   :widths: 5, 30\n\n'

            for _, refid, _ in value['files']:
                node = self.nodes[refid]

                rst += '   ' + node.name + '_ , '

                if (node.brief_description and
                        not node.brief_description.isspace()):
                    rst += node.brief_description.strip()

                rst += '\n'

            rst += '\n'
            for _, refid, _ in value['files']:
                node = self.nodes[refid]

                rst += '.. _' + node.name + ': ' + node.refid + '.html\n'

            filename = 'generated_rtd_%s.rst' % key
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(rst)

            generated_files.append(
                (filename, self.meta_info[key]['full_name']))

        rst = '\n\n.. toctree::\n   :maxdepth: 1\n   :caption: Contents:\n\n'

        for file, full_name in generated_files:
            rst += '   ' + full_name + '<' + file.replace('.rst', '') + '>\n'

        with open(os.path.join(
                    output_dir, 'generated_rtd_pages.txt'),
                  'w') as f:
            f.write(rst)

    class Node:
        def __init__(self, refid, name, xml_dir):
            self.refid = refid
            self.name = name

            self.xml_dir = xml_dir

            self.location = None

            self.brief_description = None

            self.parent = None
            self.children = []
            self.found_family = False

            self.node_xml_root = None

            self._construct_xml_root()
            self.find_location()
            self.find_brief_description()

        def _construct_xml_root(self):
            node_xml_tree = ET.parse(
                os.path.join(self.xml_dir, self.refid + '.xml'))

            node_xml_root = node_xml_tree.getroot()
            self.node_xml_root = node_xml_root.find('compounddef')

        def find_location(self, avoids=['.cxx']):
            if self.node_xml_root is None:
                raise ValueError("The Node's xml tree root has to be set!")

            location = self.node_xml_root.find('location')
            if location is not None:
                location = location.get('file')
                for txt in avoids:
                    if txt in location:
                        return
                self.location = location

        def find_brief_description(self):
            if self.node_xml_root is None:
                raise ValueError("The Node's xml tree root has to be set!")

            briefdescription = self.node_xml_root.find('briefdescription')

            brief_description = ""
            for text in briefdescription.itertext():
                brief_description += text

            self.brief_description = brief_description

        def find_family(self, nodes):
            if self.found_family:
                return

            if self.node_xml_root is None:
                raise ValueError("The Node's xml tree root has to be set!")

            parent = self.node_xml_root.find('basecompoundref')
            if parent is not None and 'refid' in parent.attrib:
                self.parent = nodes[parent.get('refid')]

            children = []
            for child_element in self.node_xml_root.findall(
                    'derivedcompoundref'):
                if 'refid' in child_element.attrib:
                    child = nodes[child_element.get('refid')]
                    children.append(child)

            self.children = children

            self.found_family = True


def main():
    teca_tree = TECA_Tree()


if __name__ == '__main__':
    main()
