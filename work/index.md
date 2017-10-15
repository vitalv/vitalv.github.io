---
layout: default
title: Vital Vialas Portfolio
---

## Projects and stuff


<!-- <p>Here is a list of selected pieces of work and projects that I have done, either for fun or at work</p> -->



{% for project in site.projects %}
*
{: .project_ul } <a class="project-link" href="{{ project.url }}" title="{{ project.title }}">{{ project.title }}</a>
	+ {% include icon-github.html username=site.github_username %} [{{project.github-repo-name}}]({{project.github-url}})



{% endfor %}

