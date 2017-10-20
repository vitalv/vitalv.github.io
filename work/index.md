---
layout: default
title: Vital Vialas Portfolio
---

## Projects and stuff


<!-- <p>Here is a list of selected pieces of work and projects that I have done, either for fun or at work</p> -->



{% for project in site.projects %}
* [{{project.title}}]({{project.url}}){: .project-link}
  * {% include icon-github.html username=site.github_username %} [{{project.github-repo-name}}]({{project.github-url}}){: target="_blank"}
  {: .project_github} 
{% endfor %}

