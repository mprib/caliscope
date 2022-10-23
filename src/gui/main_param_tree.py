# This is a place to jsut experiment with pyqt to see how I might be able to use
# it in my own applications

# A primary interest is being able to interact with an underlying confing.toml
# file to set parameters and drive the overall flow of the interaction
# ------------------------------------------------------------------------

"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types

"""

import pyqtgraph as pg
# `makeAllParamTypes` creates several parameters from a dictionary of config specs.
# This contains information about the options for each parameter so they can be directly
# inserted into the example parameter tree. To create your own parameters, simply follow
# the guidelines demonstrated by other parameters created here.
from pyqtgraph.examples._buildParamTypes import makeAllParamTypes
from pyqtgraph.Qt import QtGui, QtWidgets

app = pg.mkQApp("Multicamera Calibration")
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree



params = [
    {'name': 'Charuco', 'type': 'group', 'children': 
    [
        {'name': 'Launch Builder', 'type': 'action'},
       
    ]},

    {'name': 'Custom context menu', 'type': 'group', 'children': 
    [
        {'name': 'List contextMenu', 'type': 'float', 'value': 0, 'context': 
        [
            'menu1',
            'menu2'
        ]},
        {'name': 'Dict contextMenu', 'type': 'float', 'value': 0, 'context': {
            'changeName': 'Title',
            'internal': 'What the user sees',
        }},
    ]},
]

## Create tree of Parameter objects
p = Parameter.create(name='params', type='group', children=params)

## If anything changes in the tree, print a message
def change(param, changes):
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')
    
p.sigTreeStateChanged.connect(change)


def valueChanging(param, value):
    print("Value changing (not finalized): %s %s" % (param, value))
    
# Too lazy for recursion:
for child in p.children():
    child.sigValueChanging.connect(valueChanging)
    for ch2 in child.children():
        ch2.sigValueChanging.connect(valueChanging)
        

def save():
    global state
    state = p.saveState()

def restore():
    global state
    add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
    rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
    p.restoreState(state, addChildren=add, removeChildren=rem)

def test_function():
    print("helloworld")

p.param('Charuco', 'Launch Builder').sigActivated.connect(test_function)
# p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)


## Create two ParameterTree widgets, both accessing the same data
t = ParameterTree()
t.setParameters(p, showTop=False)
t.setWindowTitle('pyqtgraph example: Parameter Tree')

win = QtWidgets.QWidget()

layout = QtWidgets.QGridLayout()
win.setLayout(layout)
layout.addWidget(QtWidgets.QLabel("This is a label at the top."), 0,  0, 1, 2)
layout.addWidget(t, 1, 0, 1, 1)
win.show()

## test save/restore
state = p.saveState()
p.restoreState(state)
compareState = p.saveState()
assert pg.eq(compareState, state)

if __name__ == '__main__':

    pg.exec()
