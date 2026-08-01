from typing import Dict, Optional, Any
import os


class DefaultConfig:
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 1.0
    SLEEP_TIME: float = 0.5
    RATIONAL_COUNT: int = 30


class ModelConfig:
    def __init__(self, name: str, api_key_env: str, base_url: Optional[str] = None):
        self.name: str = name
        self.api_key_env: str = api_key_env
        self.base_url: Optional[str] = base_url


class ModelsConfig:
    MODELS: Dict[str, ModelConfig] = {
        "gpt-4o": ModelConfig("gpt-4o", "OPENAI_API_KEY"),
        "deepseek-reasoner": ModelConfig("deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com")
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        return cls.MODELS.get(model_name)

    @classmethod
    def get_supported_models(cls) -> list:
        return list(cls.MODELS.keys())


class PromptTemplateConfig:

    MAIN_TEMPLATE: str = """
    Generate detailed medical rationale for the diagnosis ("Diagnosis:" {{depression_status}}) based on the patient description. 
    These rationales should be the crucial cue for the diagnosis. Pretend that you don't know the diagnosis ("Diagnosis:" {{depression_status}}).
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    Also, based on DSM-5 diagnostic criteria for a major depressive episode.
    {{dsm5_criteria}}
    Diagnosis: {{depression_status}}
    Be concise but thorough. Keep your response under 500 tokens (about 375 words).
    """

    template_for_experiment_2: str = """
    Could you provide a summary of the main points concerning the mental health of the interviewee from the interview?
    {{transcript}}
    Be concise but thorough. Keep your response under 500 tokens (about 375 words).
    """


    GENDER_DSM_TEMPLATE: str = """
    You are a doctor. Diagnose the patient based on the information your colleague gave you.
    Answer me with only either "depressed" or "non-depressed".
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    Also, based on DSM-5 diagnostic criteria for a major depressive episode.
    {{dsm5_criteria}}
    """


    GENDER_DSM_TEMPLATE2: str = """
    You are a doctor. First generate detailed medical rationale, then diagnose the patient based on the information your colleague gave you.
    Your response must have the following structure:
    1. First, write a thorough medical rationale analyzing the patient's symptoms and mental state.
    2. Then, at the very end of your response, clearly state your diagnosis by writing "DIAGNOSIS: depressed" or "DIAGNOSIS: non-depressed".
    
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    
    Also, based on DSM-5 diagnostic criteria for a major depressive episode:
    {{dsm5_criteria}}
    
    Be concise but thorough in your rationale. Keep your total response under 500 tokens (about 375 words).
    Remember to first provide your rationale analysis, and only at the very end state your diagnosis.
    """

    TRANSCRIPT_DEPRESSION_TEMPLATE: str = """
    Generate detailed medical rationale for the diagnosis ("Diagnosis:" depressed) based on the patient description. 
    These rationales should be the crucial cue for the diagnosis. Pretend that you don't know the diagnosis ("Diagnosis:" depressed).
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    Also, based on DSM-5 diagnostic criteria for a major depressive episode.
    {{dsm5_criteria}}
    Diagnosis: depressed
    Be concise but thorough. Keep your response under 500 tokens (about 375 words).
    """

    template_for_experiment_4: str = """
    Could you provide a summary of the main points concerning the mental health of the interviewee from the interview?
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    Also, consider DSM-5 diagnostic criteria for a major depressive episode:
    {{dsm5_criteria}}
    Be concise but thorough. Keep your response under 500 tokens (about 375 words).
    Do not make any explicit diagnostic conclusions or assessments.
    """

    template_for_experiment_4_with_ptsd: str = """
    Could you provide a summary of the main points concerning the mental health of the interviewee from the interview?
    Patient Description: Here is a transcript of the {{gender}} patient's clinical interview, containing only the patient's utterances.
    {{transcript}}
    Also, consider DSM-5 diagnostic criteria for post-traumatic stress disorder (PTSD):
    {{dsm5_ptsd}}
    Be concise but thorough. Keep your response under 500 tokens (about 375 words).
    Do not make any explicit diagnostic conclusions or assessments.
    """

    DSM5_CRITERIA: str = """
    A. Five (or more) of the following symptoms have been present during the same 2-week period and represent a change from previous functioning; at least one of the symptoms is either (1) depressed mood or (2) loss of interest or pleasure.
    *Note: Do not include symptoms that are clearly attributable to another medical condition.
    1. Depressed mood most of the day, nearly every day, as indicated by either subjective report(e.g., feels sad, empty, hopeless) or observation made by others (e.g., appears tearful).
    2. Markedly diminished interest or pleasure in all, or almost all, activities most of the day, nearly every day (as indicated by subjective account or observation).
    3. Significant weight loss when not dieting or weight gain (e.g., change of more than 5% of body weight in a month), or decrease or increase in appetite nearly every day.
    4. Insomnia or hypersomnia nearly every day.
    5. Psychomotor agitation or retardation nearly every day (observable by others, not merely subjective feelings of restlessness or being slowed down).
    6. Fatigue or loss of energy nearly every day.
    7. Feelings of worthlessness or excessive or inappropriate guilt (which may be delusional) nearly every day (not merely self-reproach or guilt about being sick).
    8. Diminished ability to think or concentrate, or indecisiveness, nearly every day (either by subjective account or as observed by others).
    9. Recurrent thoughts of death (not just fear of dying), recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide.
    B. The symptoms cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.
    C. The episode is not attributable to the physiological effects of a substance or to another medical condition.
    * Note: Criteria A through C represent a major depressive episode.
    D. The occurrence of the major depressive episode is not better explained by schizoaffective disorder, schizophrenia, schizophreniform disorder, delusional disorder, or other specified and unspecified schizophrenia spectrum and other psychotic disorders.
    E. There has never been a manic episode or a hypomanic episode.
    * Note: This exclusion does not apply if all of the manic-like or hypomanic-like episodes are substance-induced or are attributable to the physiological effects of another medical condition.
    """

    DSM5_PTSD: str = """
    A. Exposure to actual or threatened death, serious injury, or sexual violence in one (or more) of the following ways:
    1. Directly experiencing the traumatic event(s).
    2. Witnessing, in person, the event(s) as it occurred to others.
    3. Learning that the traumatic event(s) occurred to a close family member or close friend. In cases of actual or threatened death of a family member or friend, the event(s) must have been violent or accidental.
    4. Experiencing repeated or extreme exposure to aversive details of the traumatic event(s) (e.g., first responders collecting human remains; police officers repeatedly exposed to details of child abuse). Note: Criterion A4 does not apply to exposure through electronic media, television, movies, or pictures, unless this exposure is work related.
    B. Presence of one (or more) of the following intrusion symptoms associated with the traumatic event(s), beginning after the traumatic event(s) occurred:
    1. Recurrent, involuntary, and intrusive distressing memories of the traumatic event(s). Note: In children older than 6 years, repetitive play may occur in which themes or aspects of the traumatic event(s) are expressed.
    2. Recurrent distressing dreams in which the content and/or affect of the dream are related to the traumatic event(s). Note: In children, there may be frightening dreams without recognizable content.
    3. Dissociative reactions (e.g., flashbacks) in which the individual feels or acts as if the traumatic event(s) were recurring. (Such reactions may occur on a continuum, with the most extreme expression being a complete loss of awareness of present surroundings.) Note: In children, trauma-specific reenactment may occur in play.
    4. Intense or prolonged psychological distress at exposure to internal or external cues that symbolize or resemble an aspect of the traumatic event(s).
    5. Marked physiological reactions to internal or external cues that symbolize or resemble an aspect of the traumatic event(s).
    C. Persistent avoidance of stimuli associated with the traumatic event(s), beginning after the traumatic event(s) occurred, as evidenced by one or both of the following:
    1. Avoidance of or efforts to avoid distressing memories, thoughts, or feelings about or closely associated with the traumatic event(s).
    2. Avoidance of or efforts to avoid external reminders (people, places, conversations, activities, objects, situations) that arouse distressing memories, thoughts, or feelings about or closely associated with the traumatic event(s).
    D. Negative alterations in cognitions and mood associated with the traumatic event(s), beginning or worsening after the traumatic event(s) occurred, as evidenced by two (or more) of the following:
    1. Inability to remember an important aspect of the traumatic event(s) (typically due to dissociative amnesia, and not to other factors such as head injury, alcohol, or drugs).
    2. Persistent and exaggerated negative beliefs or expectations about oneself, others, or the world (e.g., "I am bad," "No one can be trusted," "The world is completely dangerous," "My whole nervous system is permanently ruined").
    3. Persistent, distorted cognitions about the cause or consequences of the traumatic event(s) that lead the individual to blame himself/herself or others.
    4. Persistent negative emotional state (e.g., fear, horror, anger, guilt, or shame).
    5. Markedly diminished interest or participation in significant activities.
    6. Feelings of detachment or estrangement from others.
    7. Persistent inability to experience positive emotions (e.g., inability to experience happiness, satisfaction, or loving feelings).
    E. Marked alterations in arousal and reactivity associated with the traumatic event(s), beginning or worsening after the traumatic event(s) occurred, as evidenced by two (or more) of the following:
    1. Irritable behavior and angry outbursts (with little or no provocation), typically expressed as verbal or physical aggression toward people or objects.
    2. Reckless or self-destructive behavior.
    3. Hypervigilance.
    4. Exaggerated startle response.
    5. Problems with concentration.
    6. Sleep disturbance (e.g., difficulty falling or staying asleep or restless sleep).
    F. Duration of the disturbance (Criteria B, C, D and E) is more than 1 month.
    G. The disturbance causes clinically significant distress or impairment in social, occupational, or other important areas of functioning.
    H. The disturbance is not attributable to the physiological effects of a substance (e.g., medication, alcohol) or another medical condition.
    Specify whether:
    With dissociative symptoms: The individual's symptoms meet the criteria for posttraumatic stress disorder, and in addition, in response to the stressor, the individual experiences persistent or recurrent symptoms of either of the following:
    1. Depersonalization: Persistent or recurrent experiences of feeling detached from, and as if one were an outside observer of, one's mental processes or body (e.g., feeling as though one were in a dream; feeling a sense of unreality of self or body or of time moving slowly).
    2. Derealization: Persistent or recurrent experiences of unreality of surroundings (e.g., the world around the individual is experienced as unreal, dreamlike, distant, or distorted). Note: To use this subtype, the dissociative symptoms must not be attributable to the physiological effects of a substance (e.g., blackouts, behavior during alcohol intoxication) or another medical condition (e.g., complex partial seizures).
    Specify whether:
    With delayed expression: If the full diagnostic criteria are not met until at least 6 months after the event (although the onset and expression of some symptoms may be immediate).
    """


class PathConfig:
    DEFAULT_DATASET_FOLDER: str = 'data/'
    DEFAULT_INPUT_FILE: str = 'results/e_daic_transcript_test.csv'
    DEFAULT_OUTPUT_FILE: str = 'e_daic_transcript_test.csv'

    @classmethod
    def get_default_input_path(cls, env_dataset_folder: Optional[str] = None) -> str:
        base_folder = env_dataset_folder or cls.DEFAULT_DATASET_FOLDER
        return f"{base_folder}/{cls.DEFAULT_INPUT_FILE}"

    @classmethod
    def get_default_output_path(cls, env_dataset_folder: Optional[str] = None, template_type: str = "main") -> str:
        file_name = os.path.splitext(cls.DEFAULT_OUTPUT_FILE)[0]
        file_ext = os.path.splitext(cls.DEFAULT_OUTPUT_FILE)[1]
        output_file = f"{file_name}_{template_type}{file_ext}"

        base_folder = env_dataset_folder or cls.DEFAULT_DATASET_FOLDER
        results_folder = f"{base_folder}/results"
        os.makedirs(results_folder, exist_ok=True)

        return f"{results_folder}/{output_file}"


class RunConfig:
    def __init__(
        self,
        start_id: int,
        end_id: int,
        input_path: str,
        output_path: str,
        model_name: str,
        rational_count: int,
        template_type: str = "main"
    ):
        self.start_id: int = start_id
        self.end_id: int = end_id
        self.input_path: str = input_path
        self.output_path: str = output_path
        self.model_name: str = model_name
        self.rational_count: int = rational_count
        self.template_type: str = template_type

    def __str__(self) -> str:
        return (
            f"RunConfig(start_id={self.start_id}, end_id={self.end_id}, "
            f"model_name={self.model_name}, rational_count={self.rational_count}, template_type={self.template_type})"
        )
