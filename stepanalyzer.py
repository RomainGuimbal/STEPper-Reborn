#!/usr/bin/env python
#
# Copyright 2020 Doug Blanding (dblanding@gmail.com)
#
# The latest  version of this file can be found at:
# https://github.com/dblanding/step-analyzer
#
# stepanalyzer is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# stepanalyzer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# if not, write to the Free Software Foundation, Inc.
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
"""A tool which examines the hierarchical structure of a TDocStd_Document
containing CAD data in OCAF format, either loaded directly or read from a
STEP file. The structure is presented as an indented text outline."""

# Modified 2024 Tommi Hyppänen same license

from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application_GetApplication
from OCP.XCAFDoc import XCAFDoc_DocumentTool


class StepAnalyzer:
    """A class that analyzes the structure of an OCAF document."""

    def __init__(self, filename=None):
        """Supply one or the other: document or STEP filename."""
        assert filename is not None

        self.uid = 1
        self.indent = 0
        self.output = ""
        self.fname = filename
        self.doc = self.read_file(filename)

    def read_file(self, fname):
        """Read STEP file and return <TDocStd_Document>."""

        # Create the application, empty document and shape_tool
        doc = TDocStd_Document("STEP")
        app = XCAFApp_Application_GetApplication()
        app.NewDocument("MDTV-XCAF", doc)

        # Read file and return populated doc
        step_reader = STEPCAFControl_Reader()
        step_reader.SetColorMode(True)
        step_reader.SetLayerMode(True)
        step_reader.SetNameMode(True)
        step_reader.SetMatMode(True)
        status = step_reader.ReadFile(fname)
        if status == IFSelect_RetDone:
            step_reader.Transfer(doc)
            print("Transferred")
        else:
            print("Transfer failed")
            exit(1)

        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())

        return doc

    def dump(self):
        """Return assembly structure in indented outline form.

        Format of lines:
        Component Name [entry] => Referred Label Name [entry]
        Components are shown indented w/r/t line above."""

        print(">DUMP")

        if self.fname:
            self.output += f"Assembly structure of file: {self.fname}\n\n"
        else:
            self.output += "Assembly structure of doc:\n\n"
        self.indent = 0

        # Find root label of step doc
        labels = TDF_LabelSequence()
        self.shape_tool.GetFreeShapes(labels)
        print(f"Empty: {labels.IsEmpty()}")
        nbr = labels.Length()
        print("Analyzer roots:", nbr)

        rootlabel = labels.Value(1)  # First label at root

        # Get information from root label
        name = rootlabel.GetLabelName()
        entry = rootlabel.EntryDumpToString()
        is_assy = self.shape_tool.IsAssembly_s(rootlabel)
        if is_assy:
            # If 1st label at root holds an assembly, it is the Top Assy.
            # Through this label, the entire assembly is accessible.
            # There is no need to explicitly examine other labels at root.
            self.output += f"{self.uid}\t[{entry}] {name}\t"
            self.uid += 1
            self.indent += 2
            top_comps = TDF_LabelSequence()  # Components of Top Assy
            subchilds = False
            is_assy = self.shape_tool.GetComponents_s(rootlabel, top_comps, subchilds)
            self.output += f"Number of labels at root = {nbr}\n"
            if top_comps.Length():
                self.find_components(top_comps)
        return self.output

    def find_components(self, comps):
        """Discover components from comps (LabelSequence) of an assembly.

        Components of an assembly are, by definition, references which refer
        to either a shape or another assembly. Components are essentially
        'instances' of the referred shape or assembly, and carry a location
        vector specifing the location of the referred shape or assembly.
        """
        for j in range(comps.Length()):
            c_label = comps.Value(j + 1)  # component label <class 'TDF_Label'>
            c_name = c_label.GetLabelName()
            c_entry = c_label.EntryDumpToString()
            ref_label = TDF_Label()  # label of referred shape (or assembly)
            is_ref = self.shape_tool.GetReferredShape(c_label, ref_label)
            if is_ref:  # just in case all components are not references
                ref_entry = ref_label.EntryDumpToString()
                ref_name = ref_label.GetLabelName()
                indent = "  " * self.indent
                self.output += f"{self.uid}{indent}[{c_entry}] {c_name}"
                self.output += f" => [{ref_entry}] {ref_name}\n"
                self.uid += 1
                if self.shape_tool.IsAssembly_s(ref_label):
                    self.indent += 1
                    ref_comps = TDF_LabelSequence()  # Components of Assy
                    subchilds = False
                    _ = self.shape_tool.GetComponents_s(ref_label, ref_comps, subchilds)
                    if ref_comps.Length():
                        self.find_components(ref_comps)

        self.indent -= 1


if __name__ == "__main__":
    SA = StepAnalyzer(filename="tests/as1-oc-214.step")
    print(SA.dump())

#     SA2 = StepAnalyzer(filename="step/as1_pe_203.stp")
#     print(SA2.dump())
