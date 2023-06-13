---
layout: page
title: A hierarchical tree map in D3.js
description: A D3js zoomable, hierarchical tree map representing proteins grouped by categories
img: assets/img/treemap.png
importance: 1
category: work
---

Every square represents a functional group of proteins in the cell (see description on bottom right).

Click on a square to zoom in. Click on the top orange bar to zoom out. See description on mouse hover.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include d3-zoomable-treemap.html %}
    </div>
</div>



The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}
```html
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.html path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
```
{% endraw %}
