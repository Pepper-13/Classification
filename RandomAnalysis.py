#Main imports
import shapefile
from collections import OrderedDict
import xlrd
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib as mpl
import matplotlib.patches as mpatches


#Files
data_in = {
    '2014': {
        'names': ('John Tory', 'Doug Ford', 'Olivia Chow'),
        'ids': ('TORY JOHN', 'FORD DOUG', 'CHOW OLIVIA'),
        'file_shape': 'subdivisions2014/VOTING_SUBDIVISION_2014_WGS84',
        'file_res': 'results2014/MAYOR.xls'
        },
    '2010': {
        'names': ('Rob Ford', 'George Smitherman', 'Joe Pantalone'),
        'ids': ('FORD ROB', 'SMITHERMAN GEORGE', 'PANTALONE JOE'),
        'file_shape': 'voting_subdivision_2010_wgs84/VOTING_SUBDIVISION_2010_WGS84',
        'file_res': '2010_results/2010_Toronto_Poll_by_Poll_Mayor.xls'
        },
    '2006': {
        'names': ('David Miller', 'Jane Pitfield', 'Stephen LeDrew'),
        'ids': ('MILLER DAVID', 'PITFIELD JANE', 'LEDREW STEPHEN'),
        'file_shape': 'voting_subdivision_2006_wgs84/VOTING_SUBDIVISION_2006_WGS84',
        'file_res': '2006_results/2006 Results/2006_Toronto_Poll_by_Poll_Mayor.xls'
        }
    }

