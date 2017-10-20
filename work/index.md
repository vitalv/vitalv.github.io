---
layout: default
title: Vital Vialas - Work
---

## Projects and stuff


Here is a list of selected work and projects that I have not yet completely disavowed:


{% for project in site.projects %}
* [{{project.title}}]({{project.url}}){: .project-link}
  * {% include icon-github.html username=site.github_username %} [{{project.github-repo-name}}]({{project.github-url}}){: target="_blank"}
  {: .project_github} 
{% endfor %}

