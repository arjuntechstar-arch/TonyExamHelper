import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';
import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';

interface HealthResponse {
  status: 'ok';
  service: 'api';
  timestamp: string;
}

interface QuestionTemplate {
  id: string;
  _id?: string;
  name: string;
  question_type: string;
  pattern: string;
  supported_difficulties: string[];
  supported_bloom_levels: string[];
  required_fields?: string[];
  marks?: number;
  sections?: PatternSection[];
  total_marks?: number;
}

interface PatternSection {
  question_type: string;
  pattern: string;
  count: number;
  marks: number;
}

interface UserResponse {
  id: string;
  email: string;
  display_name: string;
  roles: string[];
}

interface LoginResponse {
  access_token: string;
  token_type: string;
}

interface PracticeQuestion {
  question_id: string;
  question_text: string;
  options: { key: string; text: string }[];
  difficulty: string;
}

interface PracticePayload {
  practice_test_id: string;
  status: string;
  duration_minutes: number;
  questions: PracticeQuestion[];
}

interface PracticeResult {
  percentage: number;
  correct_count: number;
  total_questions: number;
  question_results: {
    question_id: string;
    selected_answer: string | null;
    is_correct: boolean;
    explanation: string | null;
  }[];
}

interface GeneratedQuestion {
  question_text: string;
  options: { key: string; text: string }[];
  correct_answer: string | null;
  explanation: string;
  difficulty: string;
  bloom_level: string;
  sources: { chunk_id: string; page: number }[];
  marks?: number;
  question_type?: string;
  pattern?: string;
}

interface GenerationRunStatus {
  status: string;
  stage: string;
  message: string;
  logs: string[];
  result: GeneratedQuestion[] | null;
  error: string | null;
}

interface ReviewQuestion {
  id: string;
  _id?: string;
  question_text: string;
  options: { key: string; text: string }[];
  correct_answer: string | null;
  explanation: string;
  difficulty: string;
  bloom_level: string;
  review_status: string;
  review_note: string | null;
}

interface ApprovedQuestion extends ReviewQuestion {
  subject_id: string | null;
}

interface MaterialResponse {
  id: string;
  _id?: string;
  filename: string;
  status: string;
  size_bytes: number;
}

interface Subject {
  id: string;
  _id?: string;
  code: string;
  name: string;
}

interface QuestionBank {
  id: string;
  _id?: string;
  name: string;
  subject_id: string;
  question_ids: string[];
  approval_status: string;
}

interface ModelPaper {
  id: string;
  _id?: string;
  name: string;
  subject_id: string;
  question_bank_id: string;
  question_ids: string[];
  question_count: number;
  publication_status: string;
}

interface StudentAnalytics {
  student_id: string;
  total_attempts: number;
  total_questions: number;
  correct_answers: number;
  average_percentage: number;
  weak_topics: { topic_id: string; accuracy: number; attempts: number; correct_answers: number }[];
}

interface AnalyticsBreakdown {
  topic_id?: string;
  difficulty?: string;
  attempts: number;
  correct_answers: number;
  accuracy: number;
}

function normalizeQuestion<T extends { id: string; _id?: string }>(question: T): T {
  return { ...question, id: question.id || question._id || '' };
}

function normalizeId<T extends { id: string; _id?: string }>(item: T): T {
  return { ...item, id: item.id || item._id || '' };
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly http = inject(HttpClient);
  protected readonly activeView = signal('Overview');
  protected readonly currentUser = signal<UserResponse | null>(null);
  protected readonly loginOpen = signal(false);
  protected readonly loginEmail = signal('');
  protected readonly loginPassword = signal('');
  protected readonly loginMessage = signal('');
  protected readonly loginState = signal<'ready' | 'submitting'>('ready');
  protected readonly practiceSubject = signal('computer-science');
  protected readonly practiceCount = signal(10);
  protected readonly practiceDifficulty = signal('mixed');
  protected readonly practiceMessage = signal(
    'Choose a subject and configure a short practice set.',
  );
  protected readonly practiceState = signal<'ready' | 'starting' | 'started' | 'error'>('ready');
  protected readonly practiceId = signal<string | null>(null);
  protected readonly practiceQuestions = signal<PracticeQuestion[]>([]);
  protected readonly practiceAnswers = signal<Record<string, string>>({});
  protected readonly practiceIndex = signal(0);
  protected readonly practiceResult = signal<PracticeResult | null>(null);
  protected readonly apiStatus = signal<'checking' | 'online' | 'offline'>('checking');
  protected readonly apiMessage = signal('Checking the API health endpoint...');
  protected readonly generationTemplateId = signal('');
  protected readonly generationQuery = signal('');
  protected readonly generationDifficulty = signal('Medium');
  protected readonly generationBloom = signal('Understand');
  protected readonly generationCount = signal(3);
  protected readonly generatedQuestions = signal<GeneratedQuestion[]>([]);
  protected readonly generationMessage = signal(
    'Upload and index a book, select a paper pattern, choose difficulty, then generate.',
  );
  protected readonly generationStage = signal('idle');
  protected readonly generationLogs = signal<string[]>([]);
  protected readonly generationState = signal<'ready' | 'generating' | 'saving' | 'done' | 'error'>(
    'ready',
  );
  protected readonly generationTemplates = signal<QuestionTemplate[]>([]);
  protected readonly templateState = signal<'idle' | 'loading' | 'loaded' | 'error'>('idle');
  protected readonly patternName = signal('');
  protected readonly patternTotalMarks = signal(10);
  protected readonly patternSections = signal<PatternSection[]>([
    { question_type: 'MCQ', pattern: 'Direct Concept', count: 10, marks: 1 },
  ]);
  protected readonly patternState = signal<'ready' | 'saving' | 'done' | 'error'>('ready');
  protected readonly patternMessage = signal('Create a pattern before generating questions.');
  protected readonly reviewQuestions = signal<ReviewQuestion[]>([]);
  protected readonly reviewState = signal<'idle' | 'loading' | 'ready' | 'error'>('idle');
  protected readonly reviewMessage = signal('Review draft questions before adding them to a bank.');
  protected readonly approvedQuestions = signal<ApprovedQuestion[]>([]);
  protected readonly selectedQuestionIds = signal<string[]>([]);
  protected readonly bankName = signal('Computer Science Question Bank');
  protected readonly bankSubject = signal('');
  protected readonly bankMessage = signal('Select approved questions to create a reusable bank.');
  protected readonly bankCreateState = signal<'ready' | 'loading' | 'creating' | 'done' | 'error'>(
    'ready',
  );
  protected readonly materialSubject = signal('');
  protected readonly materialFile = signal<File | null>(null);
  protected readonly materialFileState = signal<'idle' | 'checking' | 'ready' | 'error'>('idle');
  protected readonly materialState = signal<
    'ready' | 'uploading' | 'processing' | 'done' | 'error'
  >('ready');
  protected readonly materialMessage = signal('Upload PDF, DOCX, PPTX, or TXT study material.');
  protected readonly materialError = signal('');
  protected readonly materialProgress = signal(0);
  protected readonly materialStage = signal<
    | 'idle'
    | 'validating'
    | 'uploading'
    | 'extracting'
    | 'chunking'
    | 'indexing'
    | 'complete'
    | 'failed'
  >('idle');
  protected readonly uploadedMaterial = signal<MaterialResponse | null>(null);
  protected readonly subjects = signal<Subject[]>([]);
  protected readonly subjectCode = signal('');
  protected readonly subjectName = signal('');
  protected readonly subjectState = signal<'idle' | 'loading' | 'creating' | 'error'>('idle');
  protected readonly subjectMessage = signal('No subjects found. Create one to upload material.');
  protected readonly dashboardState = signal<'idle' | 'loading' | 'loaded' | 'error'>('idle');
  protected readonly dashboardMessage = signal('Your workspace summary is loading.');
  protected readonly dashboardDraftCount = signal(0);
  protected readonly dashboardBankCount = signal(0);
  protected readonly dashboardPracticeScore = signal<number | null>(null);
  protected readonly questionBanks = signal<QuestionBank[]>([]);
  protected readonly bankState = signal<'idle' | 'loading' | 'ready' | 'error'>('idle');
  protected readonly paperName = signal('Midterm assessment');
  protected readonly paperSubject = signal('');
  protected readonly paperBankId = signal('');
  protected readonly paperCount = signal(10);
  protected readonly paperState = signal<'ready' | 'creating' | 'publishing' | 'done' | 'error'>(
    'ready',
  );
  protected readonly paperMessage = signal(
    'Select an approved question bank to create a model paper.',
  );
  protected readonly modelPaper = signal<ModelPaper | null>(null);
  protected readonly analyticsState = signal<'idle' | 'loading' | 'loaded' | 'error'>('idle');
  protected readonly analyticsMessage = signal('Sign in to view practice performance.');
  protected readonly studentAnalytics = signal<StudentAnalytics | null>(null);
  protected readonly topicAnalytics = signal<AnalyticsBreakdown[]>([]);
  protected readonly difficultyAnalytics = signal<AnalyticsBreakdown[]>([]);

  constructor() {
    void this.checkApiHealth();
    void this.loadCurrentUser();
  }

  private async loadCurrentUser(): Promise<void> {
    const token = localStorage.getItem('atlas_exam_token');
    if (!token) {
      return;
    }
    try {
      const user = await firstValueFrom(
        this.http.get<UserResponse>('/api/auth/me', { headers: this.authHeaders(token) }),
      );
      this.currentUser.set(user);
      void this.loadSubjects();
      void this.loadDashboard();
    } catch {
      localStorage.removeItem('atlas_exam_token');
    }
  }

  private authHeaders(token = localStorage.getItem('atlas_exam_token')): HttpHeaders {
    return token ? new HttpHeaders({ Authorization: `Bearer ${token}` }) : new HttpHeaders();
  }

  private async checkApiHealth(): Promise<void> {
    try {
      const health = await firstValueFrom(this.http.get<HealthResponse>('/api/health'));
      if (health.status !== 'ok' || health.service !== 'api') {
        throw new Error('Unexpected health response');
      }
      this.apiStatus.set('online');
      this.apiMessage.set('FastAPI foundation is responding.');
    } catch {
      this.apiStatus.set('offline');
      this.apiMessage.set('The API is not reachable. Start the backend to continue.');
    }
  }

  private async loadDashboard(): Promise<void> {
    const user = this.currentUser();
    if (!user) {
      this.dashboardState.set('error');
      this.dashboardMessage.set('Sign in to load live workspace data.');
      return;
    }
    this.dashboardState.set('loading');
    try {
      if (user.roles.some((role) => role === 'faculty' || role === 'admin')) {
        const [drafts, banks] = await Promise.all([
          firstValueFrom(
            this.http.get<ReviewQuestion[]>('/api/questions/review?review_status=draft', {
              headers: this.authHeaders(),
            }),
          ),
          firstValueFrom(
            this.http.get<QuestionBank[]>('/api/question-bank', { headers: this.authHeaders() }),
          ),
        ]);
        this.dashboardDraftCount.set(drafts.length);
        this.dashboardBankCount.set(banks.length);
      } else {
        const overview = await firstValueFrom(
          this.http.get<StudentAnalytics>(
            `/api/analytics/student?student_id=${encodeURIComponent(user.id)}`,
            {
              headers: this.authHeaders(),
            },
          ),
        );
        this.dashboardPracticeScore.set(overview.average_percentage);
      }
      this.dashboardState.set('loaded');
      this.dashboardMessage.set('Live data from your connected workspace.');
    } catch {
      this.dashboardState.set('error');
      this.dashboardMessage.set('Some workspace data could not be loaded.');
    }
  }

  protected selectView(view: string): void {
    this.activeView.set(view);
    if (view === 'Overview' && this.dashboardState() === 'idle') {
      void this.loadDashboard();
    }
    if (view === 'Question paper') {
      if (this.subjectState() === 'idle') void this.loadSubjects();
      if (this.currentUser() && this.templateState() === 'idle')
        void this.loadGenerationTemplates();
    }
    if (view === 'Generate' && this.templateState() === 'idle') {
      void this.loadGenerationTemplates();
    }
    if (view === 'Patterns' && this.templateState() === 'idle') {
      void this.loadGenerationTemplates();
    }
    if (view === 'Question bank' && this.reviewState() === 'idle') {
      void this.loadReviewQuestions();
    }
    if (view === 'Model papers' && this.bankState() === 'idle') {
      void this.loadQuestionBanks();
    }
    if (view === 'Materials' && this.subjectState() === 'idle') {
      void this.loadSubjects();
    }
    if (view === 'Practice' && this.currentUser() && this.subjectState() === 'idle') {
      void this.loadSubjects();
    }
    if (view === 'Analytics' && this.analyticsState() === 'idle') {
      void this.loadAnalytics();
    }
  }

  private async loadAnalytics(): Promise<void> {
    const user = this.currentUser();
    if (!user) {
      this.analyticsState.set('error');
      this.analyticsMessage.set('Sign in to view practice performance.');
      return;
    }
    this.analyticsState.set('loading');
    try {
      const headers = { headers: this.authHeaders() };
      const [overview, topics, difficulties] = await Promise.all([
        firstValueFrom(
          this.http.get<StudentAnalytics>(
            `/api/analytics/student?student_id=${encodeURIComponent(user.id)}`,
            headers,
          ),
        ),
        firstValueFrom(
          this.http.get<AnalyticsBreakdown[]>(
            `/api/analytics/student/topics?student_id=${encodeURIComponent(user.id)}`,
            headers,
          ),
        ),
        firstValueFrom(
          this.http.get<AnalyticsBreakdown[]>(
            `/api/analytics/student/difficulty?student_id=${encodeURIComponent(user.id)}`,
            headers,
          ),
        ),
      ]);
      this.studentAnalytics.set(overview);
      this.topicAnalytics.set(topics);
      this.difficultyAnalytics.set(difficulties);
      this.analyticsState.set('loaded');
      this.analyticsMessage.set('Performance is calculated from submitted practice answers.');
    } catch {
      this.analyticsState.set('error');
      this.analyticsMessage.set(
        'Analytics could not be loaded. Try again after completing a practice test.',
      );
    }
  }

  private async loadSubjects(): Promise<void> {
    this.subjectState.set('loading');
    try {
      const response = await firstValueFrom(
        this.http.get<Subject[]>('/api/subjects', { headers: this.authHeaders() }),
      );
      const subjects = response.map((subject) => ({
        ...subject,
        id: subject.id || subject._id || '',
      }));
      this.subjects.set(subjects);
      if (!this.materialSubject() && subjects.length) {
        this.selectMaterialSubject(subjects[0].id);
      }
      if (
        !this.practiceSubject() ||
        !subjects.some((subject) => subject.id === this.practiceSubject())
      ) {
        this.practiceSubject.set(subjects[0]?.id ?? '');
      }
      this.subjectState.set('idle');
      this.subjectMessage.set(
        subjects.length
          ? 'Select the subject for this material.'
          : 'No subjects found. Create one to upload material.',
      );
    } catch (error: unknown) {
      this.subjectState.set('error');
      const status =
        typeof error === 'object' && error !== null && 'status' in error
          ? (error as { status: number }).status
          : 0;
      this.subjectMessage.set(
        status === 403
          ? 'Your account does not have faculty or admin permission to manage subjects.'
          : 'Sign in with a faculty or admin account to load subjects.',
      );
    }
  }

  protected async createSubject(): Promise<void> {
    if (!this.subjectCode() || !this.subjectName()) {
      this.subjectState.set('error');
      this.subjectMessage.set('Enter both a subject code and subject name.');
      return;
    }
    this.subjectState.set('creating');
    try {
      const response = await firstValueFrom(
        this.http.post<Subject>(
          '/api/subjects',
          { code: this.subjectCode(), name: this.subjectName() },
          { headers: this.authHeaders() },
        ),
      );
      const subject = {
        ...response,
        id: response.id || response._id || '',
      };
      this.subjects.update((subjects) => [...subjects, subject]);
      this.selectMaterialSubject(subject.id);
      this.subjectCode.set('');
      this.subjectName.set('');
      this.subjectState.set('idle');
      this.subjectMessage.set(`${subject.code} is ready for material upload.`);
    } catch {
      this.subjectState.set('error');
      this.subjectMessage.set('Subject could not be created. Check the code and permissions.');
    }
  }

  private async loadQuestionBanks(): Promise<void> {
    this.bankState.set('loading');
    try {
      const banks = await firstValueFrom(
        this.http.get<QuestionBank[]>('/api/question-bank', { headers: this.authHeaders() }),
      );
      const approved = banks
        .map((bank) => normalizeId(bank))
        .filter((bank) => bank.approval_status === 'approved');
      this.questionBanks.set(approved);
      if (!this.paperBankId() && approved.length) {
        this.paperBankId.set(approved[0].id);
        this.paperSubject.set(approved[0].subject_id);
      }
      this.bankState.set('ready');
    } catch {
      this.bankState.set('error');
      this.paperMessage.set('Sign in with a faculty or admin account to load approved banks.');
    }
  }

  protected selectPaperBank(bankId: string): void {
    this.paperBankId.set(bankId);
    const bank = this.questionBanks().find((item) => item.id === bankId);
    if (bank) this.paperSubject.set(bank.subject_id);
  }

  protected async createModelPaper(): Promise<void> {
    if (!this.paperBankId() || !this.paperSubject()) {
      this.paperState.set('error');
      this.paperMessage.set('Select an approved question bank first.');
      return;
    }
    this.paperState.set('creating');
    this.paperMessage.set('Selecting questions from the approved bank...');
    try {
      const paper = await firstValueFrom(
        this.http.post<ModelPaper>(
          '/api/model-papers',
          {
            name: this.paperName(),
            subject_id: this.paperSubject(),
            question_bank_id: this.paperBankId(),
            question_count: this.paperCount(),
          },
          { headers: this.authHeaders() },
        ),
      );
      this.modelPaper.set(normalizeId(paper));
      this.paperState.set('done');
      this.paperMessage.set(`Draft paper created with ${paper.question_count} questions.`);
    } catch {
      this.paperState.set('error');
      this.paperMessage.set(
        'The paper could not be created. Check bank approval and question count.',
      );
    }
  }

  protected async publishModelPaper(): Promise<void> {
    const paper = this.modelPaper();
    if (!paper) return;
    this.paperState.set('publishing');
    try {
      const published = await firstValueFrom(
        this.http.post<ModelPaper>(`/api/model-papers/${paper.id}/publish`, null, {
          headers: this.authHeaders(),
        }),
      );
      this.modelPaper.set(published);
      this.paperState.set('done');
      this.paperMessage.set('Model paper published successfully.');
    } catch {
      this.paperState.set('error');
      this.paperMessage.set('The paper could not be published.');
    }
  }

  protected async loadReviewQuestions(): Promise<void> {
    this.reviewState.set('loading');
    try {
      const questions = await firstValueFrom(
        this.http.get<ReviewQuestion[]>('/api/questions/review?review_status=draft', {
          headers: this.authHeaders(),
        }),
      );
      this.reviewQuestions.set(questions.map((question) => normalizeQuestion(question)));
      this.reviewState.set('ready');
      this.reviewMessage.set(`${questions.length} draft question(s) require faculty review.`);
    } catch {
      this.reviewState.set('error');
      this.reviewMessage.set('Sign in with a faculty or admin account to review questions.');
    }
  }

  protected toggleQuestion(question: ApprovedQuestion): void {
    this.selectedQuestionIds.update((ids) =>
      ids.includes(question.id) ? ids.filter((id) => id !== question.id) : [...ids, question.id],
    );
    if (!this.bankSubject() && question.subject_id) this.bankSubject.set(question.subject_id);
  }

  protected async loadApprovedQuestions(): Promise<void> {
    this.bankCreateState.set('loading');
    try {
      const questions = await firstValueFrom(
        this.http.get<ApprovedQuestion[]>('/api/questions/review?review_status=approved', {
          headers: this.authHeaders(),
        }),
      );
      this.approvedQuestions.set(questions.map((question) => normalizeQuestion(question)));
      this.bankCreateState.set('ready');
      this.bankMessage.set(`${questions.length} approved question(s) available.`);
    } catch {
      this.bankCreateState.set('error');
      this.bankMessage.set('Sign in with a faculty or admin account to load approved questions.');
    }
  }

  protected async createQuestionBank(): Promise<void> {
    if (!this.bankName() || !this.bankSubject() || !this.selectedQuestionIds().length) {
      this.bankCreateState.set('error');
      this.bankMessage.set('Enter a bank name, subject ID, and select at least one question.');
      return;
    }
    this.bankCreateState.set('creating');
    try {
      const bank = await firstValueFrom(
        this.http.post<QuestionBank>(
          '/api/question-bank',
          {
            name: this.bankName(),
            subject_id: this.bankSubject(),
            question_ids: this.selectedQuestionIds(),
          },
          { headers: this.authHeaders() },
        ),
      );
      const normalizedBank = normalizeId(bank);
      await firstValueFrom(
        this.http.post(`/api/question-bank/${normalizedBank.id}/approve`, null, {
          headers: this.authHeaders(),
        }),
      );
      this.bankCreateState.set('done');
      this.bankMessage.set(
        `Question bank created and approved with ${normalizedBank.question_ids.length} question(s).`,
      );
      this.selectedQuestionIds.set([]);
    } catch {
      this.bankCreateState.set('error');
      this.bankMessage.set(
        'The question bank could not be created. Confirm all selected questions share the subject.',
      );
    }
  }

  protected async reviewQuestion(
    question: ReviewQuestion,
    decision: 'approve' | 'reject',
  ): Promise<void> {
    try {
      await firstValueFrom(
        this.http.post(`/api/questions/${question.id}/${decision}`, null, {
          headers: this.authHeaders(),
        }),
      );
      this.reviewQuestions.update((questions) =>
        questions.filter((item) => item.id !== question.id),
      );
      this.reviewMessage.set(decision === 'approve' ? 'Question approved.' : 'Question rejected.');
    } catch {
      this.reviewMessage.set('The review action could not be completed.');
    }
  }

  protected selectMaterialFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.materialFile.set(file);
    this.materialError.set('');
    if (!file) {
      this.materialFileState.set('idle');
      return;
    }

    this.materialFileState.set('checking');
    this.materialMessage.set(`Preparing ${file.name} (${this.formatBytes(file.size)})...`);
    const reader = new FileReader();
    reader.onload = () => {
      this.materialFileState.set('ready');
      this.materialMessage.set(`${file.name} is ready to upload.`);
    };
    reader.onerror = () => {
      this.materialFileState.set('error');
      this.materialError.set('The selected file could not be read. Choose it again.');
      this.materialMessage.set('File preparation failed.');
    };
    reader.readAsArrayBuffer(file);
  }

  protected selectMaterialSubject(subjectId: string): void {
    this.materialSubject.set(subjectId);
    this.materialError.set('');
    if (this.materialState() === 'error') {
      this.materialState.set('ready');
      this.materialStage.set('idle');
      this.materialProgress.set(0);
    }
  }

  protected async uploadMaterial(): Promise<boolean> {
    const file = this.materialFile();
    if (!this.materialSubject() && this.subjects().length === 1) {
      this.materialSubject.set(this.subjects()[0].id);
    }
    const subjectId = this.materialSubject();
    if (!subjectId || !file || this.materialFileState() !== 'ready') {
      this.materialState.set('error');
      this.materialStage.set('failed');
      const missing = !subjectId
        ? 'Select a subject before starting.'
        : !file
          ? 'Choose a file before starting.'
          : 'Wait for the file to finish loading before starting.';
      this.materialError.set(missing);
      this.materialMessage.set(missing);
      return false;
    }
    this.materialError.set('');
    this.materialProgress.set(5);
    this.materialStage.set('validating');
    this.materialState.set('uploading');
    this.materialMessage.set(`Validating ${file.name} (${this.formatBytes(file.size)})...`);
    const form = new FormData();
    form.append('subject_id', subjectId);
    form.append('file', file);
    try {
      this.materialProgress.set(20);
      this.materialStage.set('uploading');
      this.materialMessage.set('Uploading file to the backend...');
      const response = await firstValueFrom(
        this.http.post<MaterialResponse>('/api/materials/upload', form, {
          headers: this.authHeaders(),
        }),
      );
      const material = {
        ...response,
        id: response.id || response._id || '',
      };
      if (!material.id) {
        throw new Error('The upload response did not include a material ID.');
      }
      this.uploadedMaterial.set(material);
      this.materialState.set('processing');
      this.materialProgress.set(55);
      this.materialStage.set('extracting');
      this.materialMessage.set('Upload complete. Extracting text from the document...');
      await new Promise((resolve) => setTimeout(resolve, 250));
      this.materialProgress.set(75);
      this.materialStage.set('chunking');
      this.materialMessage.set('Creating retrieval chunks and saving metadata...');
      const result = await firstValueFrom(
        this.http.post<{ chunk_count: number }>(`/api/materials/${material.id}/process`, null, {
          headers: this.authHeaders(),
        }),
      );
      this.materialProgress.set(85);
      this.materialStage.set('indexing');
      this.materialMessage.set(
        `Created ${result.chunk_count} chunk(s). Indexing them for grounded retrieval...`,
      );
      const indexResult = await firstValueFrom(
        this.http.post<{ indexed_chunks: number; embedding_model: string }>(
          `/api/retrieval/materials/${material.id}/index`,
          null,
          { headers: this.authHeaders() },
        ),
      );
      this.materialState.set('done');
      this.materialProgress.set(100);
      this.materialStage.set('complete');
      this.materialMessage.set(
        `Material ready: ${indexResult.indexed_chunks} chunk(s) indexed with ${indexResult.embedding_model}.`,
      );
      return true;
    } catch (error: unknown) {
      this.materialState.set('error');
      this.materialProgress.set(0);
      this.materialStage.set('failed');
      const message = this.extractErrorMessage(error);
      this.materialError.set(message);
      this.materialMessage.set(message);
      return false;
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  private extractErrorMessage(
    error: unknown,
    fallback = 'The backend could not complete this operation.',
  ): string {
    if (typeof error === 'object' && error !== null && 'error' in error) {
      const responseError = (error as { error?: { message?: string; detail?: string } | string })
        .error;
      if (typeof responseError === 'string') return responseError;
      return responseError?.message ?? responseError?.detail ?? fallback;
    }
    return fallback;
  }

  private async loadGenerationTemplates(): Promise<void> {
    this.templateState.set('loading');
    try {
      const response = await firstValueFrom(
        this.http.get<QuestionTemplate[]>('/api/templates', { headers: this.authHeaders() }),
      );
      const templates = response.map((template) => ({
        ...template,
        id: template.id || template._id || '',
      }));
      this.generationTemplates.set(templates);
      if (!this.generationTemplateId() && templates.length) {
        this.generationTemplateId.set(templates[0].id);
        this.generationDifficulty.set(templates[0].supported_difficulties[0] ?? 'Medium');
        this.generationBloom.set(templates[0].supported_bloom_levels[0] ?? 'Understand');
      }

      this.templateState.set('loaded');
    } catch {
      this.templateState.set('error');
    }
  }

  protected async createPattern(): Promise<void> {
    const user = this.currentUser();
    if (!user || !user.roles.some((role) => role === 'faculty' || role === 'admin')) {
      this.patternState.set('error');
      this.patternMessage.set('Sign in with a faculty or admin account before creating a pattern.');
      return;
    }
    if (!this.patternName().trim()) {
      this.patternState.set('error');
      this.patternMessage.set('Enter a pattern name.');
      return;
    }
    const total = this.patternSections().reduce(
      (sum, section) => sum + section.count * section.marks,
      0,
    );
    if (total !== this.patternTotalMarks()) {
      this.patternState.set('error');
      this.patternMessage.set(
        `Section marks total ${total}, but the paper total is ${this.patternTotalMarks()}.`,
      );
      return;
    }
    this.patternState.set('saving');
    try {
      const created = await firstValueFrom(
        this.http.post<QuestionTemplate>(
          '/api/templates',
          {
            name: this.patternName().trim(),
            question_type: this.patternSections()[0].question_type,
            pattern: this.patternSections()[0].pattern,
            required_fields: ['question_text', 'explanation'],
            supported_difficulties: ['Easy', 'Medium', 'Hard'],
            supported_bloom_levels: [
              'Remember',
              'Understand',
              'Apply',
              'Analyze',
              'Evaluate',
              'Create',
            ],
            version: '1.0',
            marks: this.patternSections()[0].marks,
            sections: this.patternSections(),
            total_marks: this.patternTotalMarks(),
          },
          { headers: this.authHeaders() },
        ),
      );
      this.generationTemplates.update((templates) => [
        ...templates,
        { ...created, id: created.id || created._id || '' },
      ]);
      this.patternState.set('done');
      this.patternMessage.set(`${this.patternName()} is ready in the Generate page.`);
      this.patternName.set('');
    } catch (error: unknown) {
      this.patternState.set('error');
      this.patternMessage.set(this.extractErrorMessage(error, 'Pattern could not be created.'));
    }
  }

  protected updatePatternSection(index: number, field: keyof PatternSection, value: string): void {
    this.patternSections.update((sections) =>
      sections.map((section, sectionIndex) =>
        sectionIndex === index
          ? { ...section, [field]: field === 'count' || field === 'marks' ? Number(value) : value }
          : section,
      ),
    );
  }

  protected addPatternSection(): void {
    this.patternSections.update((sections) => [
      ...sections,
      { question_type: 'Descriptive', pattern: 'Long Answer', count: 1, marks: 5 },
    ]);
  }

  protected removePatternSection(index: number): void {
    if (this.patternSections().length === 1) return;
    this.patternSections.update((sections) =>
      sections.filter((_, sectionIndex) => sectionIndex !== index),
    );
  }

  protected selectGenerationTemplate(templateId: string): void {
    this.generationTemplateId.set(templateId);
    const template = this.generationTemplates().find((item) => item.id === templateId);
    if (template) {
      this.generationDifficulty.set(template.supported_difficulties[0] ?? 'Medium');
      this.generationBloom.set(template.supported_bloom_levels[0] ?? 'Understand');
    }
  }

  protected async startPractice(): Promise<void> {
    this.practiceState.set('starting');
    this.practiceMessage.set('Preparing your practice set...');
    try {
      const test = await firstValueFrom(
        this.http.post<PracticePayload>(
          '/api/practice/start',
          {
            subject_id: this.practiceSubject(),
            question_count: this.practiceCount(),
            difficulty: this.practiceDifficulty() === 'mixed' ? null : this.practiceDifficulty(),
          },
          { headers: this.authHeaders() },
        ),
      );
      const payload = await firstValueFrom(
        this.http.get<PracticePayload>(`/api/practice/${test.practice_test_id}`, {
          headers: this.authHeaders(),
        }),
      );
      this.practiceId.set(payload.practice_test_id);
      this.practiceQuestions.set(payload.questions);
      this.practiceAnswers.set({});
      this.practiceIndex.set(0);
      this.practiceState.set('started');
      this.practiceMessage.set('Practice set is ready.');
    } catch {
      this.practiceState.set('error');
      this.practiceMessage.set('Sign in as a student to start a practice set.');
    }
  }

  protected currentPracticeQuestion(): PracticeQuestion | undefined {
    return this.practiceQuestions()[this.practiceIndex()];
  }

  protected chooseAnswer(answer: string): void {
    const question = this.currentPracticeQuestion();
    if (!question) return;
    this.practiceAnswers.update((answers) => ({ ...answers, [question.question_id]: answer }));
  }

  protected nextPracticeQuestion(): void {
    if (this.practiceIndex() < this.practiceQuestions().length - 1)
      this.practiceIndex.update((index) => index + 1);
  }

  protected previousPracticeQuestion(): void {
    if (this.practiceIndex() > 0) this.practiceIndex.update((index) => index - 1);
  }

  protected async submitPractice(): Promise<void> {
    const id = this.practiceId();
    if (!id) return;
    this.practiceMessage.set('Scoring your answers...');
    try {
      const result = await firstValueFrom(
        this.http.post<PracticeResult>(
          `/api/practice/${id}/submit`,
          { answers: this.practiceAnswers() },
          { headers: this.authHeaders() },
        ),
      );
      this.practiceResult.set(result);
      this.practiceMessage.set('Practice complete. Review your explanations below.');
    } catch {
      this.practiceState.set('error');
      this.practiceMessage.set('Your answers could not be submitted. Please try again.');
    }
  }

  protected resultFor(questionId: string): PracticeResult['question_results'][number] | undefined {
    return this.practiceResult()?.question_results.find(
      (result) => result.question_id === questionId,
    );
  }

  protected openLogin(): void {
    this.loginOpen.set(true);
    this.loginMessage.set('');
  }

  protected closeLogin(): void {
    if (this.loginState() === 'ready') {
      this.loginOpen.set(false);
    }
  }

  protected async submitLogin(): Promise<void> {
    this.loginState.set('submitting');
    this.loginMessage.set('Signing you in...');
    try {
      const response = await firstValueFrom(
        this.http.post<LoginResponse>('/api/auth/login', {
          email: this.loginEmail(),
          password: this.loginPassword(),
        }),
      );
      localStorage.setItem('atlas_exam_token', response.access_token);
      const user = await firstValueFrom(
        this.http.get<UserResponse>('/api/auth/me', {
          headers: this.authHeaders(response.access_token),
        }),
      );
      this.currentUser.set(user);
      if (user.roles.some((role) => role === 'faculty' || role === 'admin')) {
        void this.loadSubjects();
      }
      this.loginPassword.set('');
      this.loginOpen.set(false);
      this.dashboardState.set('idle');
      void this.loadDashboard();
    } catch {
      this.loginMessage.set('Unable to sign in. Check your email and password.');
    } finally {
      this.loginState.set('ready');
    }
  }

  protected logout(): void {
    localStorage.removeItem('atlas_exam_token');
    this.currentUser.set(null);
  }

  protected async generateQuestions(): Promise<void> {
    if (!this.currentUser()) {
      this.generationState.set('error');
      this.generationMessage.set('Sign in as faculty or admin to generate a question paper.');
      this.openLogin();
      return;
    }
    if (this.materialState() !== 'done') {
      this.generationState.set('generating');
      this.generationMessage.set('Uploading and indexing your book before generation...');
      const indexed = await this.uploadMaterial();
      if (!indexed) {
        this.generationState.set('error');
        this.generationMessage.set(this.materialMessage());
        return;
      }
    }
    this.generationState.set('generating');
    this.generationMessage.set('Starting generation worker...');
    this.generationStage.set('starting');
    this.generationLogs.set([]);
    try {
      const run = await firstValueFrom(
        this.http.post<{ id: string }>(
          '/api/questions/generate/paper/start',
          {
            template_id: this.generationTemplateId(),
            difficulty: this.generationDifficulty(),
            bloom_level: this.generationBloom(),
            subject_id: this.materialSubject() || null,
            top_k: 5,
          },
          { headers: this.authHeaders() },
        ),
      );
      let status: GenerationRunStatus;
      do {
        await new Promise((resolve) => setTimeout(resolve, 500));
        status = await firstValueFrom(
          this.http.get<GenerationRunStatus>(`/api/questions/generate/runs/${run.id}`, {
            headers: this.authHeaders(),
          }),
        );
        this.generationStage.set(status.stage);
        this.generationMessage.set(status.message);
        this.generationLogs.set(status.logs);
      } while (status.status === 'queued' || status.status === 'running');
      if (status.status !== 'completed' || !status.result) {
        throw new Error(status.error || status.message || 'Generation failed.');
      }
      const questions = status.result;
      this.generatedQuestions.set(questions);
      this.generationState.set('done');
      this.generationMessage.set(
        `${questions.length} candidate question(s) generated. Review before saving.`,
      );
      this.activeView.set('Generated questions');
    } catch (error: unknown) {
      this.generationState.set('error');
      this.generationMessage.set(
        this.extractErrorMessage(
          error,
          'Generation failed. Upload and index material, then select a valid pattern and source topic.',
        ),
      );
    }
  }

  protected async saveGeneratedQuestion(question: GeneratedQuestion): Promise<void> {
    this.generationState.set('saving');
    try {
      await firstValueFrom(
        this.http.post(
          '/api/questions',
          {
            question,
            question_type: question.question_type ?? 'MCQ',
            pattern: question.pattern ?? 'Direct Concept',
            template_id: this.generationTemplateId(),
            marks: question.marks ?? 1,
          },
          { headers: this.authHeaders() },
        ),
      );
      this.generationState.set('done');
      this.generationMessage.set('Question saved as a draft for faculty review.');
    } catch {
      this.generationState.set('error');
      this.generationMessage.set(
        'Question could not be saved. Sign in with a faculty or admin account.',
      );
    }
  }
}
