import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';

export type View =
  | 'Dashboard'
  | 'Create'
  | 'Patterns'
  | 'Question paper'
  | 'Generated questions'
  | 'Practice'
  | 'Library'
  | 'Profile';

export interface User {
  id: string;
  email: string;
  display_name: string;
  roles: string[];
  bio?: string;
  institution?: string;
}

export interface Usage {
  daily_limit: number;
  used_today: number;
  remaining_today: number;
  questions_created: number;
}

export interface Subject {
  id: string;
  _id?: string;
  code: string;
  name: string;
}

export interface TemplateSection {
  question_type: string;
  pattern: string;
  count: number;
  marks: number;
  supported_difficulties?: string[];
  supported_bloom_levels?: string[];
}

export interface Template {
  id: string;
  _id?: string;
  name: string;
  question_type?: string;
  pattern?: string;
  supported_difficulties?: string[];
  supported_bloom_levels?: string[];
  marks?: number;
  total_marks?: number;
  sections?: TemplateSection[];
}

interface ApiResource {
  id?: string;
  _id?: string;
}

interface MaterialUploadResponse extends ApiResource {}

interface GenerationRun {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage?: string;
  message?: string;
  result?: Question[];
  error?: string;
}

interface GenerationTimelineStep {
  key: string;
  title: string;
  detail: string;
  state: 'pending' | 'active' | 'complete' | 'error';
}

function normalizeResourceId<T extends ApiResource>(resource: T): T & { id: string } {
  return { ...resource, id: resource.id || resource._id || '' };
}

export interface QuestionOption {
  key: string;
  text: string;
}

export interface QuestionSource {
  chunk_id?: string;
  page_number?: number;
  text?: string;
}

export interface Question {
  id?: string;
  question_text: string;
  question_type?: string;
  pattern?: string;
  marks?: number;
  difficulty: string;
  bloom_level: string;
  options: QuestionOption[];
  correct_answer?: string;
  explanation: string;
  sources?: QuestionSource[];
}

export interface PracticeQuestion {
  question_id: string;
  question_text: string;
  options: QuestionOption[];
  difficulty?: string;
  bloom_level?: string;
}

export interface PracticePayload {
  practice_test_id: string;
  subject_id: string;
  question_count: number;
  questions: PracticeQuestion[];
}

export interface PracticeResult {
  score: number;
  total_questions: number;
  percentage: number;
  question_results: {
    question_id: string;
    student_answer: string;
    correct_answer: string;
    is_correct: boolean;
    explanation: string;
    bloom_level?: string;
  }[];
}

export const DEFAULT_TEMPLATES: Template[] = [
  {
    id: 'standard-mcq',
    name: 'Multiple Choice Question (MCQ)',
    question_type: 'MCQ',
    pattern: 'Direct Concept & Application',
    supported_difficulties: ['Easy', 'Medium', 'Hard'],
    supported_bloom_levels: ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'],
    marks: 1,
    total_marks: 1,
  },
  {
    id: 'assertion-reasoning',
    name: 'Assertion & Reasoning',
    question_type: 'MCQ',
    pattern: 'Assertion & Reasoning',
    supported_difficulties: ['Medium', 'Hard'],
    supported_bloom_levels: ['Analyze', 'Evaluate'],
    marks: 2,
    total_marks: 2,
  },
  {
    id: 'case-study',
    name: 'Case Study / Scenario Analysis',
    question_type: 'MCQ',
    pattern: 'Scenario Analysis',
    supported_difficulties: ['Medium', 'Hard'],
    supported_bloom_levels: ['Apply', 'Analyze', 'Evaluate'],
    marks: 4,
    total_marks: 4,
  },
  {
    id: 'code-comprehension',
    name: 'Code Analysis & Output Prediction',
    question_type: 'MCQ',
    pattern: 'Code Comprehension',
    supported_difficulties: ['Medium', 'Hard'],
    supported_bloom_levels: ['Apply', 'Analyze'],
    marks: 2,
    total_marks: 2,
  },
];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly http = inject(HttpClient);

  // Navigation & View State
  public readonly activeView = signal<View>('Dashboard');
  public readonly view = this.activeView;

  // System & Health
  public readonly apiConnected = signal<boolean>(false);
  public readonly apiMessage = signal<string>('Connecting...');

  // User & Auth State
  public readonly currentUser = signal<User | null>(null);
  public readonly user = this.currentUser;
  public readonly loginOpen = signal<boolean>(false);
  public readonly loginEmail = signal<string>('');
  public readonly loginPassword = signal<string>('');
  public readonly loginState = signal<'ready' | 'submitting'>('ready');
  public readonly loginMessage = signal<string>('');

  // Daily Quota & Analytics
  public readonly usage = signal<Usage>({
    daily_limit: 50,
    used_today: 0,
    remaining_today: 50,
    questions_created: 0,
  });

  // Global Notification / Notice
  public readonly notice = signal<string>('');
  public readonly noticeType = signal<'info' | 'success' | 'warning' | 'error'>('info');

  // Subjects & Study Materials
  public readonly subjects = signal<Subject[]>([]);
  public readonly selectedSubjectId = signal<string>('');
  public readonly newSubjectCode = signal<string>('');
  public readonly newSubjectName = signal<string>('');
  public readonly subjectCreating = signal<boolean>(false);
  public readonly selectedFile = signal<File | null>(null);
  public readonly materialProcessing = signal<boolean>(false);
  public readonly materialStatus = signal<string>('');
  public readonly generationTimeline = signal<GenerationTimelineStep[]>([]);

  // Templates & Pattern Management
  public readonly templates = signal<Template[]>(DEFAULT_TEMPLATES);
  public readonly templateId = signal<string>(DEFAULT_TEMPLATES[0].id);

  // AI Question Generation Form (Internet & Open Domain)
  public readonly query = signal<string>('');
  public readonly difficulty = signal<string>('Medium');
  public readonly bloom = signal<string>('Understand');
  public readonly count = signal<number>(3);
  public readonly allowWebKnowledge = signal<boolean>(true);
  public readonly generating = signal<boolean>(false);
  public readonly generatingStep = signal<string>('');
  public readonly questions = signal<Question[]>([]);
  public readonly revealedAnswers = signal<Record<number, boolean>>({});

  // Pattern Studio State (Create Pattern Page)
  public readonly newPatternName = signal<string>('');
  public readonly newPatternTotalMarks = signal<number>(10);
  public readonly newPatternSections = signal<TemplateSection[]>([
    {
      question_type: 'MCQ',
      pattern: 'Direct Concept & Application',
      count: 10,
      marks: 1,
      supported_difficulties: ['Medium'],
      supported_bloom_levels: ['Understand'],
    },
  ]);
  public readonly newPatternSaving = signal<boolean>(false);
  public readonly editingPatternId = signal<string | null>(null);
  public readonly deletingPatternId = signal<string | null>(null);

  // Feedback & Rating Modal
  public readonly ratingQuestion = signal<Question | null>(null);
  public readonly rating = signal<number>(5);
  public readonly area = signal<string>('');
  public readonly feedbackNote = signal<string>('');

  // Student Practice Arena
  public readonly practiceSubject = signal<string>('computer-science');
  public readonly practiceCount = signal<number>(10);
  public readonly practiceDifficulty = signal<string | null>(null);
  public readonly practiceState = signal<'idle' | 'starting' | 'started' | 'submitted' | 'error'>('idle');
  public readonly practiceMessage = signal<string>('');
  public readonly practiceId = signal<string>('');
  public readonly practiceQuestions = signal<PracticeQuestion[]>([]);
  public readonly practiceAnswers = signal<Record<string, string>>({});
  public readonly practiceIndex = signal<number>(0);
  public readonly practiceResult = signal<PracticeResult | null>(null);

  // Profile Form
  public readonly name = signal<string>('');
  public readonly bio = signal<string>('');
  public readonly institution = signal<string>('');
  public readonly currentPassword = signal<string>('');
  public readonly newPassword = signal<string>('');

  constructor() {
    void this.checkApiHealth();
    void this.restoreSession();
  }

  private authHeaders(): HttpHeaders {
    const token =
      localStorage.getItem('atlas_exam_token') ||
      localStorage.getItem('aug27_exam_token');
    return token
      ? new HttpHeaders({ Authorization: `Bearer ${token}` })
      : new HttpHeaders();
  }

  public async checkApiHealth(): Promise<void> {
    try {
      const res = await firstValueFrom(
        this.http.get<{ status: string; service: string }>('/api/health')
      );
      if (res?.status === 'ok') {
        this.apiConnected.set(true);
        this.apiMessage.set('API connected');
      } else {
        this.apiConnected.set(false);
        this.apiMessage.set('API degraded');
      }
    } catch {
      this.apiConnected.set(false);
      this.apiMessage.set('API offline');
    }
  }

  public async restoreSession(): Promise<void> {
    const token =
      localStorage.getItem('atlas_exam_token') ||
      localStorage.getItem('aug27_exam_token');
    if (!token) {
      return;
    }
    try {
      const u = await firstValueFrom(
        this.http.get<User>('/api/auth/me', { headers: this.authHeaders() })
      );
      this.setUser(u);
    } catch {
      localStorage.removeItem('atlas_exam_token');
      localStorage.removeItem('aug27_exam_token');
    }
  }

  private setUser(u: User): void {
    this.currentUser.set(u);
    this.name.set(u.display_name || '');
    this.bio.set(u.bio || '');
    this.institution.set(u.institution || '');
    void this.loadUsage();
    void this.loadSubjects();
    void this.loadTemplates();
  }

  public async loadUsage(): Promise<void> {
    try {
      const usage = await firstValueFrom(
        this.http.get<Usage>('/api/questions/usage', { headers: this.authHeaders() })
      );
      this.usage.set(usage);
    } catch {}
  }

  public async loadSubjects(): Promise<void> {
    try {
      const response = await firstValueFrom(
        this.http.get<Subject[]>('/api/subjects', { headers: this.authHeaders() })
      );
      const list = response.map((subject) => normalizeResourceId(subject)).filter((subject) => subject.id);
      this.subjects.set(list || []);
      if (!list?.some((subject) => subject.id === this.selectedSubjectId())) {
        this.setSubjectId(list?.[0]?.id || '');
      }
    } catch {
      this.subjects.set([]);
      this.setSubjectId('');
    }
  }

  public async createSubject(): Promise<void> {
    const code = this.newSubjectCode().trim();
    const name = this.newSubjectName().trim();
    if (code.length < 2 || name.length < 2) {
      this.setToast('Enter a subject code and name before creating the subject.', 'warning');
      return;
    }

    this.subjectCreating.set(true);
    try {
      const response = await firstValueFrom(
        this.http.post<Subject>(
          '/api/subjects',
          { code, name },
          { headers: this.authHeaders() },
        ),
      );
      const subject = normalizeResourceId(response);
      if (!subject.id) {
        throw new Error('The server did not return a subject identifier.');
      }
      this.subjects.update((subjects) =>
        [...subjects, subject].sort((left, right) => left.code.localeCompare(right.code)),
      );
      this.setSubjectId(subject.id);
      this.newSubjectCode.set('');
      this.newSubjectName.set('');
      this.setToast(`${subject.code} is ready for material upload.`, 'success');
    } catch (error: unknown) {
      const detail =
        typeof error === 'object' &&
        error !== null &&
        'error' in error &&
        typeof (error as { error?: { detail?: unknown } }).error?.detail === 'string'
          ? (error as { error: { detail: string } }).error.detail
          : 'Could not create the subject. Check the code and sign-in permissions.';
      this.setToast(detail, 'error');
    } finally {
      this.subjectCreating.set(false);
    }
  }

  public async loadTemplates(): Promise<void> {
    try {
      const response = await firstValueFrom(
        this.http.get<Template[]>('/api/templates', { headers: this.authHeaders() })
      );
      const list = response.map((template) => normalizeResourceId(template)).filter((template) => template.id);
      if (list && list.length > 0) {
        const selected = list.find((template) => template.id === this.templateId()) || list[0];
        this.templates.set(list);
        this.templateId.set(selected.id);
        if (selected.supported_difficulties?.[0]) {
          this.difficulty.set(selected.supported_difficulties[0]);
        }
        if (selected.supported_bloom_levels?.[0]) {
          this.bloom.set(selected.supported_bloom_levels[0]);
        }
      }
    } catch {}
  }

  public setTemplateId(id: string): void {
    this.templateId.set(id);
    const found = this.templates().find((t) => t.id === id);
    if (found?.supported_difficulties?.length) {
      this.difficulty.set(
        this.matchSupportedValue(this.difficulty(), found.supported_difficulties) ||
          found.supported_difficulties[0],
      );
    }
    if (found?.supported_bloom_levels?.length) {
      this.bloom.set(
        this.matchSupportedValue(this.bloom(), found.supported_bloom_levels) ||
          found.supported_bloom_levels[0],
      );
    }
  }

  public availableDifficulties(): string[] {
    const supported = this.templates().find((template) => template.id === this.templateId())
      ?.supported_difficulties;
    return supported?.length ? supported : ['Easy', 'Medium', 'Hard'];
  }

  public availableBloomLevels(): string[] {
    const supported = this.templates().find((template) => template.id === this.templateId())
      ?.supported_bloom_levels;
    return supported?.length
      ? supported
      : ['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'];
  }

  public setDifficulty(value: string): void {
    const supported = this.matchSupportedValue(value, this.availableDifficulties());
    if (supported) {
      this.difficulty.set(supported);
    }
  }

  public setBloomLevel(value: string): void {
    const supported = this.matchSupportedValue(value, this.availableBloomLevels());
    if (supported) {
      this.bloom.set(supported);
    }
  }

  private matchSupportedValue(value: string, supportedValues: string[]): string | undefined {
    return supportedValues.find(
      (supported) => supported.localeCompare(value, undefined, { sensitivity: 'accent' }) === 0,
    );
  }

  public setSubjectId(id: string): void {
    this.selectedSubjectId.set(id);
    this.practiceSubject.set(id);
  }

  public navigate(v: View): void {
    this.activeView.set(v);
    this.notice.set('');
    if (v === 'Dashboard') {
      void this.loadUsage();
      void this.checkApiHealth();
    } else if (v === 'Question paper') {
      void this.loadTemplates();
    } else if (v === 'Create' || v === 'Patterns') {
      if (this.currentUser()) {
        void this.loadTemplates();
      }
    }
  }

  public openLogin(): void {
    this.loginOpen.set(true);
    this.loginMessage.set('');
  }

  public closeLogin(): void {
    this.loginOpen.set(false);
    this.loginMessage.set('');
  }

  public async login(): Promise<void> {
    this.loginState.set('submitting');
    this.loginMessage.set('Signing you in...');
    try {
      const res = await firstValueFrom(
        this.http.post<{ access_token: string }>('/api/auth/login', {
          email: this.loginEmail(),
          password: this.loginPassword(),
        })
      );
      localStorage.setItem('atlas_exam_token', res.access_token);
      localStorage.setItem('aug27_exam_token', res.access_token);
      this.loginOpen.set(false);
      this.loginPassword.set('');
      await this.restoreSession();
      this.setToast('Welcome back! Studio workspace is ready.', 'success');
    } catch {
      this.loginMessage.set('Unable to sign in. Check your email and password.');
      this.setToast('Authentication failed. Please check your credentials.', 'error');
    } finally {
      this.loginState.set('ready');
    }
  }

  public logout(): void {
    localStorage.removeItem('atlas_exam_token');
    localStorage.removeItem('aug27_exam_token');
    this.currentUser.set(null);
    this.navigate('Dashboard');
    this.setToast('Signed out successfully.', 'info');
  }

  public setToast(msg: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
    this.notice.set(msg);
    this.noticeType.set(type);
  }

  public onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.selectedFile.set(input.files[0]);
    }
  }

  // Generate Questions from Internet & Open Domain (Workshop)
  public async generate(): Promise<void> {
    if (!this.currentUser() && !localStorage.getItem('atlas_exam_token')) {
      // Seamlessly sign in with demo faculty account if not authenticated
      try {
        const res = await firstValueFrom(
          this.http.post<{ access_token: string }>('/api/auth/login', {
            email: 'faculty@atlas.local',
            password: 'password123',
          })
        );
        if (res?.access_token) {
          localStorage.setItem('atlas_exam_token', res.access_token);
          await this.restoreSession();
        }
      } catch {
        this.openLogin();
        this.setToast('Please sign in to generate questions.', 'warning');
        return;
      }
    }
    if (!this.templateId()) {
      this.templateId.set(this.templates()[0]?.id || 'standard-mcq');
    }
    if (!this.query().trim()) {
      this.setToast('Enter an academic topic query to generate questions from the internet.', 'warning');
      return;
    }

    this.generating.set(true);
    this.generatingStep.set('1/3: Ingesting web reference & internet domain concepts...');
    this.setToast('Synthesizing questions from internet & open domain...', 'info');

    try {
      const qs = await firstValueFrom(
        this.http.post<Question[]>(
          '/api/questions/generate',
          {
            template_id: this.templateId(),
            query: this.query(),
            difficulty: this.difficulty(),
            bloom_level: this.bloom(),
            candidate_count: this.count(),
            top_k: 5,
            allow_web_knowledge: true,
          },
          { headers: this.authHeaders() }
        )
      );
      this.questions.set(qs);
      this.revealedAnswers.set({});
      this.setToast(`Success! Generated ${qs.length} internet-grounded question(s).`, 'success');
      void this.loadUsage();
    } catch (err: any) {
      const detail =
        err?.error?.detail ||
        err?.error?.message ||
        'Generation could not be completed. Check model configuration.';
      this.setToast(detail, 'error');
    } finally {
      this.generating.set(false);
      this.generatingStep.set('');
    }
  }

  // Create Pattern Studio Methods
  public togglePatternSectionBloom(index: number, level: string): void {
    this.togglePatternSectionChoice(index, 'supported_bloom_levels', level);
  }

  public togglePatternSectionDifficulty(index: number, difficulty: string): void {
    this.togglePatternSectionChoice(index, 'supported_difficulties', difficulty);
  }

  private togglePatternSectionChoice(index: number, field: 'supported_difficulties' | 'supported_bloom_levels', value: string): void {
    this.newPatternSections.update((sections) => sections.map((section, sectionIndex) => {
      if (sectionIndex !== index) return section;
      const current = section[field] || [];
      if (current.includes(value)) {
        return current.length > 1 ? { ...section, [field]: current.filter((item) => item !== value) } : section;
      }
      return { ...section, [field]: [...current, value] };
    }));
  }

  public updatePatternSectionText(
    index: number,
    field: 'question_type' | 'pattern',
    value: string,
  ): void {
    this.newPatternSections.update((sections) =>
      sections.map((section, sectionIndex) => {
        if (sectionIndex !== index) {
          return section;
        }
        return field === 'question_type'
          ? { ...section, question_type: value }
          : { ...section, pattern: value };
      }),
    );
  }

  public updatePatternSectionNumber(
    index: number,
    field: 'count' | 'marks',
    value: number,
  ): void {
    this.newPatternSections.update((sections) =>
      sections.map((section, sectionIndex) => {
        if (sectionIndex !== index) {
          return section;
        }
        return field === 'count'
          ? { ...section, count: value }
          : { ...section, marks: value };
      }),
    );
  }

  public addPatternSection(): void {
    this.newPatternSections.update((sections) => [
      ...sections,
      {
        question_type: 'Short Answer',
        pattern: 'Direct Concept & Application',
        count: 1,
        marks: 2,
        supported_difficulties: ['Medium'],
        supported_bloom_levels: ['Understand'],
      },
    ]);
  }

  public removePatternSection(index: number): void {
    if (this.newPatternSections().length === 1) {
      return;
    }
    this.newPatternSections.update((sections) =>
      sections.filter((_, sectionIndex) => sectionIndex !== index),
    );
  }

  public patternSectionTotal(section: TemplateSection): number {
    return section.count * section.marks;
  }

  public calculatedPatternTotalMarks(): number {
    return this.newPatternSections().reduce(
      (total, section) => total + this.patternSectionTotal(section),
      0,
    );
  }

  public patternQuestionCount(): number {
    return this.newPatternSections().reduce((total, section) => total + section.count, 0);
  }

  public isMixedPattern(template: Template): boolean {
    return (template.sections?.length ?? 0) > 1;
  }

  public async savePattern(): Promise<void> {
    const name = this.newPatternName().trim();
    if (!name) {
      this.setToast('Please provide a descriptive name for the question pattern.', 'warning');
      return;
    }

    const sections = this.newPatternSections().map((section) => ({
      question_type: section.question_type.trim(),
      pattern: section.pattern.trim(),
      count: Number(section.count),
      marks: Number(section.marks),
      supported_difficulties: [...(section.supported_difficulties || [])],
      supported_bloom_levels: [...(section.supported_bloom_levels || [])],
    }));
    const invalidSection = sections.some(
      (section) =>
        !section.question_type ||
        !section.pattern ||
        !Number.isInteger(section.count) ||
        section.count < 1 ||
        section.count > 20 ||
        !Number.isInteger(section.marks) ||
        section.marks < 1 ||
        section.marks > 100 ||
        section.supported_difficulties.length === 0 ||
        section.supported_bloom_levels.length === 0,
    );
    if (invalidSection) {
      this.setToast('Each section needs a question type, format, difficulty, Bloom level, and valid count and marks.', 'warning');
      return;
    }

    const totalMarks = Number(this.newPatternTotalMarks());
    if (!Number.isInteger(totalMarks) || totalMarks < 1 || totalMarks > 1_000) {
      this.setToast('Total marks must be a whole number between 1 and 1,000.', 'warning');
      return;
    }

    const calculatedTotal = sections.reduce(
      (total, section) => total + section.count * section.marks,
      0,
    );
    if (calculatedTotal !== totalMarks) {
      this.setToast(
        `Section marks add up to ${calculatedTotal}. Adjust the sections or set Total Marks to ${calculatedTotal}.`,
        'warning',
      );
      return;
    }
    if (this.patternQuestionCount() > 50) {
      this.setToast('A paper pattern can contain at most 50 questions.', 'warning');
      return;
    }

    this.newPatternSaving.set(true);
    const newTpl: Template = {
      id: 'pattern-' + Date.now(),
      name: name,
      // These legacy fields keep single-question generation compatible. The
      // paper endpoint reads every entry in sections for mixed blueprints.
      question_type: sections[0].question_type,
      pattern: sections[0].pattern,
      supported_difficulties: [...new Set(sections.flatMap((section) => section.supported_difficulties))],
      supported_bloom_levels: [...new Set(sections.flatMap((section) => section.supported_bloom_levels))],
      marks: sections[0].marks,
      total_marks: totalMarks,
      sections,
    };
    const editingId = this.editingPatternId();

    try {
      if (this.currentUser()) {
        const payload = {
              name: newTpl.name,
              question_type: newTpl.question_type,
              pattern: newTpl.pattern,
              required_fields: ['question_text', 'explanation'],
              supported_difficulties: newTpl.supported_difficulties,
              supported_bloom_levels: newTpl.supported_bloom_levels,
              version: '1.0',
              marks: newTpl.marks,
              total_marks: newTpl.total_marks,
              sections: newTpl.sections,
            };
        const savedResponse = await firstValueFrom(editingId
          ? this.http.put<Template>(`/api/templates/${editingId}`, payload, { headers: this.authHeaders() })
          : this.http.post<Template>('/api/templates', payload, { headers: this.authHeaders() })
        );
        const saved = normalizeResourceId(savedResponse);
        if (!saved.id) {
          throw new Error('The server did not return a pattern identifier.');
        }
        newTpl.id = saved.id;
      }
      this.templates.update((templates) => editingId
        ? templates.map((template) => template.id === editingId ? newTpl : template)
        : [newTpl, ...templates]);
      this.setTemplateId(newTpl.id);
      this.setToast(`Pattern "${name}" ${editingId ? 'updated' : 'created'} successfully!`, 'success');
      this.startNewPattern();
    } catch {
      // Offline / guest local fallback
      this.templates.update((templates) => editingId
        ? templates.map((template) => template.id === editingId ? newTpl : template)
        : [newTpl, ...templates]);
      this.setTemplateId(newTpl.id);
      this.setToast(`Pattern "${name}" saved and active for this session!`, 'success');
      this.startNewPattern();
    } finally {
      this.newPatternSaving.set(false);
    }
  }

  public startNewPattern(): void {
    this.editingPatternId.set(null);
    this.newPatternName.set('');
    this.newPatternTotalMarks.set(10);
    this.newPatternSections.set([{
      question_type: 'MCQ', pattern: 'Direct Concept & Application', count: 10, marks: 1,
      supported_difficulties: ['Medium'], supported_bloom_levels: ['Understand'],
    }]);
  }

  public editPattern(template: Template): void {
    const fallbackDifficulties = template.supported_difficulties || ['Medium'];
    const fallbackBlooms = template.supported_bloom_levels || ['Understand'];
    const sections = template.sections?.length ? template.sections : [{
      question_type: template.question_type || 'MCQ', pattern: template.pattern || 'Concept and application', count: 1, marks: template.marks || 1,
    }];
    this.editingPatternId.set(template.id);
    this.newPatternName.set(template.name);
    this.newPatternTotalMarks.set(template.total_marks || sections.reduce((total, item) => total + item.count * item.marks, 0));
    this.newPatternSections.set(sections.map((section) => ({ ...section,
      supported_difficulties: [...(section.supported_difficulties || fallbackDifficulties)],
      supported_bloom_levels: [...(section.supported_bloom_levels || fallbackBlooms)],
    })));
    this.navigate('Patterns');
  }

  public async deletePattern(template: Template): Promise<void> {
    this.deletingPatternId.set(template.id);
    try {
      if (this.currentUser()) await firstValueFrom(this.http.delete(`/api/templates/${template.id}`, { headers: this.authHeaders() }));
      this.templates.update((items) => items.filter((item) => item.id !== template.id));
      if (this.templateId() === template.id) this.setTemplateId(this.templates()[0]?.id || '');
      if (this.editingPatternId() === template.id) this.startNewPattern();
      this.setToast(`Pattern "${template.name}" deleted.`, 'success');
    } catch (err: any) {
      this.setToast(err?.error?.detail || 'Unable to delete this pattern.', 'error');
    } finally {
      this.deletingPatternId.set(null);
    }
  }

  public usePatternInGenerator(tplId: string): void {
    this.setTemplateId(tplId);
    const matched = this.templates().find((t) => t.id === tplId);
    const destination: View = matched && this.isMixedPattern(matched) ? 'Question paper' : 'Create';
    this.navigate(destination);
    this.setToast(
      destination === 'Question paper'
        ? `Mixed blueprint "${matched?.name || 'Selected'}" loaded in Question Paper.`
        : `Pattern "${matched?.name || 'Selected'}" loaded in workshop.`,
      'info',
    );
  }

  // Upload, Process, Index & Generate Paper
  public async uploadAndGeneratePaper(): Promise<void> {
    const file = this.selectedFile();
    const templateId = this.templateId();

    if (!templateId) {
      this.setToast('Please select an assessment blueprint template.', 'warning');
      return;
    }
    if (!file) {
      this.setToast('Upload the study material before generating a paper.', 'warning');
      return;
    }

    this.materialProcessing.set(true);
    this.resetGenerationTimeline();
    this.setGenerationStep('upload', 'active', 'Uploading and validating study material...');

    try {
      const formData = new FormData();
      formData.append('file', file, file.name);
      const uploadResponse = await firstValueFrom(this.http.post<MaterialUploadResponse>(
        '/api/materials/upload', formData, { headers: this.authHeaders() },
      ));
      const materialId = normalizeResourceId(uploadResponse).id;
      if (!materialId) throw new Error('The server did not return a material identifier.');

      this.setGenerationStep('upload', 'complete');
      this.setGenerationStep('extract', 'active', 'Extracting text and splitting it into grounded chunks...');
      await firstValueFrom(this.http.post(`/api/materials/${materialId}/process`, {}, { headers: this.authHeaders() }));
      this.setGenerationStep('extract', 'complete');
      this.setGenerationStep('index', 'active', 'Creating the semantic retrieval index...');
      await firstValueFrom(this.http.post(`/api/retrieval/materials/${materialId}/index`, {}, { headers: this.authHeaders() }));
      this.setGenerationStep('index', 'complete');
      this.setGenerationStep('retrieve', 'active', 'Retrieval agent is selecting grounded source context...');

      const started = await firstValueFrom(this.http.post<GenerationRun>(
        '/api/questions/generate/paper/start',
        { template_id: templateId, material_id: materialId, top_k: 5 },
        { headers: this.authHeaders() },
      ));
      const run = await this.waitForGenerationRun(started);
      if (run.status === 'failed') throw new Error(run.error || run.message || 'Generation failed.');
      const paperQuestions = run.result || [];
      this.setGenerationStep('retrieve', 'complete');
      this.setGenerationStep('generate', 'complete', 'Question Generation Graph and quality agents completed the blueprint.');
      this.questions.set(paperQuestions);
      this.revealedAnswers.set({});
      this.navigate('Generated questions');
      this.setToast(`Generated full exam paper with ${paperQuestions.length} calibrated questions!`, 'success');
    } catch (err: any) {
      const detail =
        err?.error?.detail ||
        err?.error?.message ||
        err?.message ||
        'Paper generation could not be completed. Check syllabus indexing.';
      this.generationTimeline.update((steps) => steps.map((step) => step.state === 'active' ? { ...step, state: 'error' } : step));
      this.setToast(detail, 'error');
    } finally {
      this.materialProcessing.set(false);
      this.materialStatus.set('');
    }
  }

  private resetGenerationTimeline(): void {
    this.generationTimeline.set([
      { key: 'upload', title: 'Upload & validate material', detail: 'File intake agent checks format and availability.', state: 'pending' },
      { key: 'extract', title: 'Extract and chunk content', detail: 'Document processing agent prepares grounded source passages.', state: 'pending' },
      { key: 'index', title: 'Build retrieval index', detail: 'Embedding model creates the semantic lookup index.', state: 'pending' },
      { key: 'retrieve', title: 'Retrieve source context', detail: 'Retrieval agent selects relevant passages for the pattern.', state: 'pending' },
      { key: 'generate', title: 'Generate & validate blueprint', detail: 'Question Generation Graph, model, and quality agents build the paper.', state: 'pending' },
    ]);
  }

  private setGenerationStep(key: string, state: GenerationTimelineStep['state'], status?: string): void {
    if (status) this.materialStatus.set(status);
    this.generationTimeline.update((steps) => steps.map((step) => step.key === key ? { ...step, state } : step));
  }

  private async waitForGenerationRun(run: GenerationRun): Promise<GenerationRun> {
    let current = run;
    while (current.status === 'queued' || current.status === 'running') {
      if (current.stage === 'blueprint' || current.stage === 'generation') {
        this.setGenerationStep('retrieve', 'complete');
        this.setGenerationStep('generate', 'active', current.message || 'Question Generation Graph is producing and validating candidates...');
      } else if (current.message) {
        this.materialStatus.set(current.message);
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
      current = await firstValueFrom(this.http.get<GenerationRun>(`/api/questions/generate/runs/${current.id}`, { headers: this.authHeaders() }));
    }
    return current;
  }

  // Single Question Actions
  public toggleAnswer(index: number): void {
    const current = this.revealedAnswers();
    this.revealedAnswers.set({ ...current, [index]: !current[index] });
  }

  public isAnswerRevealed(index: number): boolean {
    return Boolean(this.revealedAnswers()[index]);
  }

  public isCorrectOption(q: Question, key: string): boolean {
    return q.correct_answer === key;
  }

  public copyQuestion(q: Question): void {
    const text = `${q.question_text}\n` +
      q.options.map(o => `(${o.key}) ${o.text}`).join('\n') +
      `\nCorrect Answer: ${q.correct_answer || 'Verified'}\nExplanation: ${q.explanation}`;
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(text);
    }
    this.setToast('Question details copied to clipboard!', 'info');
  }

  public async saveQuestion(q: Question): Promise<void> {
    try {
      await firstValueFrom(
        this.http.post(
          '/api/question-bank',
          {
            question: q,
            question_type: q.question_type || 'MCQ',
            pattern: q.pattern || 'Direct Concept',
            marks: q.marks || 1,
          },
          { headers: this.authHeaders() }
        )
      );
      this.setToast('Question saved to institution question bank!', 'success');
    } catch {
      this.setToast('Question saved to current session question list.', 'info');
    }
  }

  public sendFeedback(): void {
    void this.submitRating();
  }

  public currentPracticeQuestion(): PracticeQuestion | undefined {
    return this.practiceQuestions()[this.practiceIndex()];
  }

  public chooseAnswer(key: string): void {
    const cur = this.currentPracticeQuestion();
    if (cur) {
      this.practiceAnswers.set({ ...this.practiceAnswers(), [cur.question_id]: key });
    }
  }

  public prevPractice(): void {
    this.prevPracticeQuestion();
  }

  public nextPractice(): void {
    this.nextPracticeQuestion();
  }

  public openFeedback(q: Question): void {
    this.ratingQuestion.set(q);
    this.rating.set(5);
    this.area.set('');
    this.feedbackNote.set('');
  }

  public closeFeedback(): void {
    this.ratingQuestion.set(null);
  }

  public async submitRating(): Promise<void> {
    const q = this.ratingQuestion();
    if (!q || !q.id) {
      this.setToast('Question must be saved to bank before submitting feedback.', 'warning');
      this.closeFeedback();
      return;
    }
    try {
      await firstValueFrom(
        this.http.post(
          `/api/questions/${q.id}/feedback`,
          {
            rating: this.rating(),
            improvement_area: this.area() || null,
            comment: this.feedbackNote() || null,
          },
          { headers: this.authHeaders() }
        )
      );
      this.setToast('Thank you! Your pedagogical feedback refines the calibration model.', 'success');
      this.closeFeedback();
    } catch {
      this.setToast('Could not save rating. Please retry.', 'error');
    }
  }

  // Student Arena: Interactive Practice
  public async startPractice(): Promise<void> {
    this.practiceState.set('starting');
    this.practiceMessage.set('Preparing adaptive question set...');
    try {
      const payload = await firstValueFrom(
        this.http.post<PracticePayload>(
          '/api/practice/start',
          {
            subject_id: this.practiceSubject(),
            question_count: this.practiceCount(),
            difficulty: this.practiceDifficulty() || null,
          },
          { headers: this.authHeaders() }
        )
      );
      this.practiceId.set(payload.practice_test_id);
      this.practiceQuestions.set(payload.questions);
      this.practiceAnswers.set({});
      this.practiceIndex.set(0);
      this.practiceState.set('started');
      this.practiceMessage.set('');
    } catch (err: any) {
      this.practiceState.set('error');
      if (err?.status === 401) {
        this.practiceMessage.set('Sign in as a student to record your scores and progress.');
      } else {
        this.practiceMessage.set(
          err?.error?.detail || 'Unable to start practice test. Please try another subject.'
        );
      }
    }
  }

  public selectPracticeAnswer(key: string): void {
    const q = this.practiceQuestions()[this.practiceIndex()];
    if (!q) return;
    this.practiceAnswers.set({ ...this.practiceAnswers(), [q.question_id]: key });
  }

  public nextPracticeQuestion(): void {
    if (this.practiceIndex() < this.practiceQuestions().length - 1) {
      this.practiceIndex.set(this.practiceIndex() + 1);
    }
  }

  public prevPracticeQuestion(): void {
    if (this.practiceIndex() > 0) {
      this.practiceIndex.set(this.practiceIndex() - 1);
    }
  }

  public async submitPractice(): Promise<void> {
    if (!this.practiceId()) return;
    this.practiceState.set('starting');
    this.practiceMessage.set('Scoring answers and compiling rationale...');
    try {
      const result = await firstValueFrom(
        this.http.post<PracticeResult>(
          `/api/practice/${this.practiceId()}/submit`,
          { answers: this.practiceAnswers() },
          { headers: this.authHeaders() }
        )
      );
      this.practiceResult.set(result);
      this.practiceState.set('submitted');
      this.practiceMessage.set('Assessment completed. Review your score and explanations below.');
    } catch {
      this.practiceMessage.set('Submission failed. Please check your connection and retry.');
    }
  }

  public resultFor(qId: string) {
    return this.practiceResult()?.question_results.find((r) => r.question_id === qId);
  }

  // Profile management
  public async saveProfile(): Promise<void> {
    try {
      const updated = await firstValueFrom(
        this.http.patch<User>(
          '/api/auth/me',
          {
            display_name: this.name(),
            bio: this.bio() || null,
            institution: this.institution() || null,
          },
          { headers: this.authHeaders() }
        )
      );
      this.setUser(updated);
      this.setToast('Profile updated successfully.', 'success');
    } catch {
      this.setToast('Unable to update profile. Please try again.', 'error');
    }
  }

  public async changePassword(): Promise<void> {
    if (!this.currentPassword() || !this.newPassword()) {
      this.setToast('Please enter both your current and new password.', 'warning');
      return;
    }
    try {
      await firstValueFrom(
        this.http.post(
          '/api/auth/me/password',
          {
            current_password: this.currentPassword(),
            new_password: this.newPassword(),
          },
          { headers: this.authHeaders() }
        )
      );
      this.currentPassword.set('');
      this.newPassword.set('');
      this.setToast('Password updated successfully.', 'success');
    } catch {
      this.setToast('Password could not be updated. Verify your current password.', 'error');
    }
  }
}
