{{/*
NVFlare server chart helpers
*/}}
{{- define "nvflare-server.name" -}}
{{- default .Chart.Name .Values.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nvflare-server.labels" -}}
app.kubernetes.io/name: {{ include "nvflare-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nvflare-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nvflare-server.name" . }}
{{- end }}

{{- define "nvflare-server.image" -}}
{{- if .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- else }}
{{- .Values.image.repository }}
{{- end }}
{{- end }}

{{- define "nvflare-server.hostPortEnabled" -}}
{{- if hasKey .Values "hostPortEnabled" }}
{{- .Values.hostPortEnabled | toString }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}

{{/*
Workspace volume mounts shared by the main server container and the
state-store migration init container.
*/}}
{{- define "nvflare-server.workspaceVolumeMounts" -}}
- name: {{ .Values.persistence.workspace.volumeName }}
  mountPath: {{ .Values.persistence.workspace.mountPath }}
{{- with .Values.workspaceConfig }}
{{- with .local }}
{{- if .configMapName }}
- name: workspace-local
  mountPath: {{ $.Values.persistence.workspace.mountPath }}/local
  readOnly: true
{{- end }}
{{- end }}
{{- with .startup }}
{{- if .secretName }}
- name: workspace-startup
  mountPath: {{ $.Values.persistence.workspace.mountPath }}/startup
  readOnly: true
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{- define "nvflare-server.tcpConfigMapEnabled" -}}
{{- if hasKey .Values "tcpConfigMapEnabled" }}
{{- .Values.tcpConfigMapEnabled | toString }}
{{- else }}
{{- "false" }}
{{- end }}
{{- end }}
