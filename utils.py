#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 23:53:03 2020

@author: nemo
"""

import pickle


def save_pickle(file, filepath):
    pickle.dump(file, open(filepath, 'wb'))
    

def load_pickle(filepath):
    f = pickle.load(open(filepath, 'rb'))
    return f