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

def dict_results(results_excel):
    '''Gives you a results dictionary for a given excel doc.  The dictionary has entries which are dictionaries themselves - one for each subdivision in the file.'''
    workbook = xlrd.open_workbook(results_excel)
    worksheets = workbook.sheet_names()
    results = {}

    for worksheet_name in worksheets:
        worksheet = workbook.sheet_by_name(worksheet_name)

        #Loop through all of the columns other than the first and last
        for col in range(1, worksheet.ncols - 2):
            #I think this subdivision tab is only in 2006.\n,
            if worksheet.cell_value(1,col) != 'Subdivision':
                #Initialize the subdivision dictionary
                dic = {}
                identifier = worksheet_name[4::].zfill(2) + str(int(worksheet.cell_value(1, col))).zfill(3)
                #Loop through all candidates
                for r in range(2, worksheet.nrows):
                    if r < worksheet.nrows - 1:
                         dic[worksheet.cell_value(r, 0)] = worksheet.cell_value(r, col)
                    else:
                         dic['ALLCANDIDATES'] = worksheet.cell_value(r, col)
                results[identifier] = dic
          
    #Uncomment to check out one
    #print results[results.keys()[0]]
    
    return results
          
def shaperead(file_shape):
    '''Reads data from a shape file and returns some vertices.'''
    sf = shapefile.Reader(file_shape)
    all_verts = []
    
    #Grabs all the vertices
    for shape in sf.shapes():
        all_verts.append([(point[0],point[1]) for point in shape.points])        
    identifiers = [rec[2] for rec in sf.records()]
    
    return (all_verts, identifiers)

def make_graph(graphname, percs, peeps, folder=''):
    '''Creates a graph, given a graphname, an array of percentages, and an array of people names (for titles)
    Add in the folder variable to save to a different folder.
    '''
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(14,7))
    colls = [PolyCollection(all_verts, array=perc, cmap=mpl.cm.coolwarm, edgecolors='none') for perc in percs]
    
    for a, col, peep in zip(ax,colls,peeps):
        col.set_clim(0,0.9)
        a.add_collection(col)
        a.autoscale_view()
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
        a.set_title(peep, fontsize=12)


