# Bioinformatics

I am currently a Bioinformatics PhD student at UCL. In this section of my blog I discuss code I write and talk about the research I conduct!

## Code optimisation
{% for post in site.posts %}
{% if post.tags contains "bioinformatics" and post.tags contains "optimisation"%}
- [{{ post.title }}]({{ post.url }})
{% endif %}
{% endfor %}
