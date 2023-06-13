---
layout: page
title: A D3js treemap with radio buttons
description:  A D3js tree map representing how protein weights change according to treatments
img: assets/img/radio_treemap.png
importance: 2
category: work
---

This here is a treemap generated using D3.js ```treemap``` and its ```resquarify``` function   
  
It was created as part of a project in which I studied the protein economy of the cell.   

The surface of the color-coded protein categories represent the mass fraction in proportion to total protein weight.   

<a href="https://gist.github.com/vitalv/6aef6fd31c6fd03a048dd3057851226c">Project on github</a>


And the radio-buttons at the bottom represent different culture conditions, 60 and 300 represent light intensities;  
glc stands for glucose added to culture medium and dcmu is a photosynthesis inhibitor. This was quite revealing!   


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include d3resquarify.html %}
