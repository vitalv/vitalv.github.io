---
layout: default
title: Vital Vialas Portfolio
---

## Projects and stuff

Here is a list of selected pieces of work and projects that I have done, either for fun or at work

{% for project in site.projects %}
* <a href="{{ project.url }}" title="{{ project.title }}">{{ project.title }}</a>
{% endfor %}

