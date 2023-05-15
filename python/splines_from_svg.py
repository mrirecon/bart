#!/usr/bin/python
# -- coding: utf-8 --

"""
Copyright 2022. Uecker Lab, University Medical Center Goettingen.

Authors:
2022 Martin Schilling (martin.schilling@med.uni-goettingen.de)
2022 Nick Scholand (scholand@tugraz.at)

DESCRIPTION :
This script takes an SVG file as an input, analyses the paths of objects,
which can consist of horizontal, vertical, diagonal and cubic spline
transformations, splits these transformations up into cubic Hermite splines
and creates a CFL file for use with the BART phantom command line tool.
"""

import os
import numpy as np
import argparse

import sys
sys.path.insert(0, os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
import cfl

DBLEVEL = 0


def read_svg(svg_input, scale_flag=True):
	"""
	Reads a given svg file to extract parameters of paths.

	:param str svg_input: File path to input svg.
	:param bool scale_coords: Boolean for scaling coordinates
	:returns: List of lists for paths. Element: [object_id, color, transforms]
	:rtype: list
	"""
	paths_list = []
	prev_key_list, points_list = [], []
	readout = False

	with open (svg_input, 'rt', encoding="utf8", errors='ignore') as input:
		for line in input:
			if "<path" in line:
				readout = True
				object_id="000"
				color = "000000"
				prev_keys=[]
				points=[]

			if readout:
				if "style=" in line:
					if "stroke:#" in line:
						color = line.split("stroke:#")[1][:6]

					if "fill:#" in line:
						color = line.split("fill:#")[1][:6]

				if "     d=" in line:
					prev_keys, points = analyse_d_string(line.split('"')[1])

				if "id=" in line:
					object_id = line.split('"')[1]

				# end of parameters
				if "/>" in line:
					readout = False
					prev_key_list.append(prev_keys)
					points_list.append(points)
					paths_list.append([object_id, color])
	input.close()

	if scale_flag:
		scale_coords(points_list, center=[0,0], norm=1.8)

	for num,(k,p) in enumerate(zip(prev_key_list,points_list)):
		if 0 != len(k):
			transforms = get_transforms(k,p)
			paths_list[num].append(transforms)
		else:
			paths_list[num].append([])

	return paths_list

def scale_coords(coords_list, center=[187.5, 125], norm=350):
	"""
	Scale coordinates to a given center with a maximal norm.

	:param list coords_list: List of lists of coordinates. Each list belongs to a series of control points.
	:param list center: Coordinates of new center
	:param int norm: Maximal value for width and height
	"""
	x_min = coords_list[0][0][0]
	x_max = coords_list[0][0][0]
	y_min = coords_list[0][0][1]
	y_max = coords_list[0][0][1]

	# determine maximal and minimal x- and y-values
	for cs in coords_list:
		for c in cs:
			x_max = c[0] if c[0] > x_max else x_max
			x_min = c[0] if c[0] < x_min else x_min
			y_max = c[1] if c[1] > y_max else y_max
			y_min = c[1] if c[1] < y_min else y_min

	# transfer values to new center
	x_trans = (x_max + x_min) / 2
	y_trans = (y_max + y_min) / 2
	# normalization factor of coordinates as ratio of norm and max([width,height])
	norm_factor = norm / max([np.abs(x_max-x_min), np.abs(y_max-y_min)])

	for cs in coords_list:
		for c in cs:
			c[0] = (c[0] - x_trans) * norm_factor + center[0]
			c[1] = (c[1] - y_trans) * norm_factor + center[1]

def try_float(string):
	# Function for trying a string for conversion to float.
	try:
		f = float(string)
		return True
	except ValueError:
		return False

def analyse_d_string(d_string):
	"""
	Analyse string of 'd' argument of path. The function returns the signal transform parameters
	and a list of coordinates of the control points.

	:param str d_string: Complete string contained in the 'd' parameter of a path.
	:returns: tuple(transform_keys, coordinates)
		WHERE
		list transform_keys is list of lower case signal characters for transformations
		list coordinates is list of absolute coordinates of control points
	"""
	content = d_string.split()
	prev_key = None
	points = []
	transf_keys = []
	cspline = []
	count = 0
	x_origin, y_origin = 0, 0
	for num,section in enumerate(content):
		if len(section.split(",")) > 1 or try_float(section):
			# keys before coordinates signal new transformation,
			# lower case for relative, upper case for absolute coordinates
			special_keys = ['c','C','m','M', 'l', 'L']

			# deal with exception, that 'm'/'M' keys may be followed by diagonal
			# transformation ('l' key) without explicit key
			if len(section.split(",")) > 1 and prev_key not in special_keys:
				prev_key = "l"

			if "c" == prev_key or "C" == prev_key:

				count += 1

				if 3 == count:
					cspline.append([cspline[1][0]+cspline[1][0]-cspline[0][0],cspline[1][1]+cspline[1][1]-cspline[0][1]])
					if "c" == prev_key:
						# relative reference point
						x_origin += float(content[num].split(",")[0])
						y_origin += float(content[num].split(",")[1])
					if "C" == prev_key:
						# absolute reference point
						x_origin = float(content[num].split(",")[0])
						y_origin = float(content[num].split(",")[1])

					# append intermediate control points
					points.append([cspline[0][0], cspline[0][1]])
					transf_keys.append(prev_key)
					points.append([cspline[1][0], cspline[1][1]])
					transf_keys.append(prev_key)

					count = 0
					cspline = []
				else:
					if "c" == prev_key:
						cspline.append([x_origin+float(content[num].split(",")[0]), y_origin+float(content[num].split(",")[1])])
					if "C" == prev_key:
						cspline.append([float(content[num].split(",")[0]), float(content[num].split(",")[1])])
			else:
				count = 0

			# start of path
			if "m" == prev_key or "M" == prev_key:
				x_origin = float(content[num].split(",")[0])
				y_origin = float(content[num].split(",")[1])

			# horizontal transformation
			if "h" == prev_key:
				x_origin += float(content[num])
			if "H" == prev_key:
				x_origin = float(content[num])

			# vertical transformation
			if "v" == prev_key:
				y_origin += float(content[num])
			if "V" == prev_key:
				y_origin = float(content[num])

			# diagonal transformation
			if "l" == prev_key:
				x_origin += float(content[num].split(",")[0])
				y_origin += float(content[num].split(",")[1])
			if "L" == prev_key:
				x_origin = float(content[num].split(",")[0])
				y_origin = float(content[num].split(",")[1])

			if 0 == count:
				points.append([x_origin, y_origin])
				transf_keys.append(prev_key.lower())
				if 'M' == prev_key:
					prev_key = 'L'
				if 'm' == prev_key:
					prev_key = 'l'
		else:
			prev_key = section

	return transf_keys, points

def controlpoints2cspline(bezier_points):
	"""
	Translate four input control points into a cubic Hermite spline format suitable for BART.

	:param list bezier_points: List of four control points in format [p1,p2,p3,p4] with p_i=[x_i,y_i]
	:returns: Parameters for cubic Hermite spline [x_parameters, y_parameters]
	:rtype: list
	"""
	bezier_cspline = [[1,-3,0,0],[0,3,0,0],[0,0,0,-3],[0,0,1,3]]
	bezier_x = [p[0] for p in bezier_points]
	bezier_y = [p[1] for p in bezier_points]
	bezier = [bezier_x, bezier_y]
	cspline = [[0,0]  for i in range(4)]

	for num, c in enumerate(bezier):
		for i in range(4):
			for j in range(4):
				cspline[i][num] += bezier_cspline[j][i] * bezier[num][j]

	cspline_x = [p[0] for p in cspline]
	cspline_y = [p[1] for p in cspline]

	return [cspline_x, cspline_y]

def get_transforms(keys, points):
	"""
	Create separate transformations from given lists of keys and coordinates.
	The transformations have the form [[x_transforms],[y_transforms]] in the cubic Hermite spline format.

	:param list keys: List of signal characters for path transformations [key1, key2, ...]
	:param list points: List of coordinates [[x1,y1], [x2,y2], ...]
	:returns: Transformations in cubic Hermite spline format
	:rtype: list
	"""
	transforms = []

	for num,(k,p) in enumerate(zip(keys,points)):

		if 'h' == k:
			transforms.append([[points[num-1][0],0.,p[0],-0.],[p[1],0.,p[1],-0.]])

		if 'v' == k:
			transforms.append([[p[0],0.,p[0],-0.],[points[num-1][1],0.,p[1],-0.]])

		if 'l' == k:
			transforms.append([[points[num-1][0],0.,p[0],-0.],[points[num-1][1],0.,p[1],-0.]])

		if num+1 < len(points) and 'c' == k:
			# non-trivial B-spline
			if 'c' == keys[num-1] and 'c' == keys[num+1]:
				keys[num+1] = None
				transforms.append(controlpoints2cspline(points[num-2:num+2]))

			# trivial B-spline
			elif 'c' != keys[num-1] and 'c' != keys[num+1]:
				transforms.append([[points[num-1][0],0.,p[0],-0.],[points[num-1][1],0.,p[1],-0.]])

	return transforms

def format_transforms(transforms, object_id, filename, output_file):
	"""
	Format transforms for insertion into /bart/src/geom/logo.c

	:param list transform: List of transformations in cubic Hermite spline format
	:param list object_id: List of object ids for indexing transformations
	:param str filename: Name of struct
	:param str output_file: File path to output text file
	"""
	total_transforms = sum([len(t) for t in transforms])
	with open (output_file, 'w', encoding="utf8", errors='ignore') as output:
		output.write("//Replace in bart/src/geom/logo.c > bart_logo and adjust bart/src/geom/logo.h\n\n")
		output.write("const double "+filename+"["+str(total_transforms)+"][2][4] = {\n")
		for num, transform in enumerate(transforms):
			output.write("\t//"+str(object_id[num])+"\n")
			for enum,t in enumerate(transform):
				x_string = str(t[0][0])+", "+str(t[0][1])+", "+str(t[0][2])+", "+str(t[0][3])
				y_string = str(t[1][0])+", "+str(t[1][1])+", "+str(t[1][2])+", "+str(t[1][3])
				# current implementation in BART, likely to change to x_string, y_string in the future
				output.write("\t{ { "+y_string+" }, { "+x_string+" } },\n")
		output.write("};\n")

def transform2polystruct(transforms, id_color, output_file):
	"""
	Create a polystruct for a given set of transformations and append it to output file.
	Can replace code in bart/src/simu/phantom.c > calc_bart

	:param list transforms: List of transformations in [[x_transforms],[y_transforms]] format
	:param list id_color: List of fill colors of individual objects
	:param str output_file: File path to output text file
	"""
	total_transforms = sum([len(t) for t in transforms])

	with open (output_file, 'a', encoding="utf8", errors='ignore') as output:
		output.write("\tint N = "+str(total_transforms)+";\n")
		output.write("\tdouble points[N * 11][2];\n")
		output.write("\n")
		output.write("\tstruct poly poly = {\n\t\tkspace,\n\t\tcoeff,\n\t\tpopts->large_sens,\n\t\t"+str(len(transforms))+",\n\t\t&(struct poly1[]){\n")
		array_position = 0
		for num, transform in enumerate(transforms):
			output.write("\t\t\t{ "+str(len(transform)*11)+" , "+str(id_color[num])+", ARRAY_SLICE(points, "+str(array_position*11)+", "+str((array_position+len(transform))*11) +") },\n")
			array_position += len(transform)
		output.write("\t\t}\n")
		output.write("\t};")
	output.close()

def assign_color_id(colors):
	"""
	Extract color IDs from hex colors

	:param list colors: List of strings representing the objects colors

	:returns: List of Integers representing the objects colors as integer (> 0 !) IDs
	:rtype: list
	"""

	color_values, color_counts = np.unique(colors, return_counts=True)

	id_color = [list(color_values).index(i)+1 for i in colors]

	return id_color

# Save geometry data in numpy array
# 	coord -> [segment, cp_set:[x,y], cp_coord] with control points (cp)
#	meta -> [path index, number of segments, color of path]
def save2cfl(new_transforms, new_colors, cfl_output):

	coord = []
	meta = []
	ind_path = 0

	for sub_array in new_transforms:

		ind_seg = 0

		for path in sub_array:

			path_array = np.array(path)
			coord.append(path_array)

			ind_seg += 1

		meta.append(np.array([ind_path, ind_seg, new_colors[ind_path]]))
		ind_path += 1

	coord = np.array(coord)
	meta = np.array(meta)
	
	if (2 <= DBLEVEL):
		print("Coord Dims:")
		print(np.shape(coord))
		print("Meta Dims:")
		print(np.shape(meta))
		print("Meta:")
		print(meta)

	cfl.writemulticfl(cfl_output, np.array([coord, meta], dtype=object))


def main(svg_input, text_output, output):
	"""
	Extract parameters of paths from SVG file and write code block into txt file,
	which is suitable for bart/src/simu/shepplogan.c > calc_bart and bart/src/geom/logo.c.

	:param str svg_input: File path to input SVG file
	:param str cfl: File path to output cfl file. Default: <svg_name>.{cfl,hdr}
	"""

	if (text_output):
		text_filename = output+".txt"

	path_objects = read_svg(svg_input)

	object_ids = [obj[0] for obj in path_objects]
	colors = [obj[1] for obj in path_objects]
	transforms = [obj[2] for obj in path_objects]

	# Sort paths by color (=: grey value in provided SVG file)

	id_color = assign_color_id(colors)

	new_colors = sorted(id_color)
	new_ids = [id for color, id in sorted(zip(id_color,object_ids))]
	new_transforms = [trans for color, trans in sorted(zip(id_color,transforms))]

	color_values, color_counts = np.unique(new_colors, return_counts=True)

	if (2 <= DBLEVEL):
		print("Distribution of colors:")
		print("Value:\t", color_values)
		print("Number:\t", color_counts)


	save2cfl(new_transforms, new_colors, output)


	if (1 <= DBLEVEL):
		print("Created files:")
		print(output+".{cfl,hdr}")


	if (text_output):

		format_transforms(new_transforms, new_ids, output, text_filename)

		transform2polystruct(new_transforms, new_colors, text_filename)

		if (1 <= DBLEVEL):
			print(output+".txt")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Script to extract control points of cubic Hermite splines from SVG file to CFL format.")

	parser.add_argument('input', type=str, help="Input SVG file")
	parser.add_argument('output', type=str, help="Output CFL filename")
	parser.add_argument('-d', '--db', default=-1, type=int, help="Specify debug value for additional information  [default: 0]")

	# Internal option for more complicated objects with multi component paths (example: BRAIN geometry)
	# Requires manual tuning and is therefore hidden for simplicity
	parser.add_argument('-t', action='store_true', help=argparse.SUPPRESS)

	args = parser.parse_args()

	if ("DEBUG_LEVEL" in os.environ):

		if (-1 != args.db):
			print("A local DEBUG_LEVEL variable exists! It will be overwritten by -d input!\n")

		DBLEVEL = int(os.environ["DEBUG_LEVEL"])

	if (-1 != args.db):
		DBLEVEL = args.db

	main(args.input, args.t, args.output)